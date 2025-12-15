#!/usr/bin/env python3
import yaml
import time
import json
import re
import ast
import os
from openai import OpenAI
from anthropic import Anthropic
from func_timeout import func_set_timeout
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import logging
from config import config
from prompts import get_instruction
from model_manager import ModelManager, get_model_manager, ModelType

@dataclass
class GPTResponse:
    """Data class for structured GPT response."""
    danger_score: float
    reason: str
    navigation: Optional[str] = None
    processing_time: float = 0.0
    model_used: str = "unknown"

class GPTInterface:
    def __init__(self, api_key_path: str = "API_KEYS.yaml") -> None:
        """
        Initialize GPT interface with model manager.
        """
        try:
            # Initialize model manager
            self.model_manager = get_model_manager(api_key_path)
            self.setup_prompts()
            logging.info("GPTInterface initialized with ModelManager")
        except Exception:
            logging.exception("Error initializing GPTInterface")
            raise

    def set_model(self, model_name: str) -> bool:
        """Set the current model using the model manager."""
        return self.model_manager.set_current_model(model_name)
    
    def get_current_model(self) -> str:
        """Get the current model name."""
        return self.model_manager.get_current_model()

    def setup_prompts(self) -> None:
        """Setup system, location, and format prompts with obstacle sensitivity."""
        # End-to-end instruction assembled from prompts.py, parameterized by sensitivity
        self.system_prompt = get_instruction(system_sensitivity=config.SYSTEM_SENSITIVITY)

        # Extra rules to bias models towards obstacle sensitivity and consistent scoring/actions
        self.obstacle_rules = (
            "Obstacle sensitivity rules:\n"
            "- Treat any object in the 'ground' region as an immediate obstacle unless clearly safe.\n"
            "- If an object's width or height exceeds ~15% of the frame in front/center, raise danger_score >= 0.7.\n"
            "- Prefer STOP when danger_score >= 0.8, especially for obstacles directly ahead.\n"
            "- When uncertain, err on the side of caution with conservative guidance.\n"
        )

        # JSON output schema guidance - clean format without comments
        self.format_prompt = (
            "Analyze the object detection data and provide a brief assessment.\n\n"
            "Output format (valid JSON only):\n"
            "{\n"
            '  "danger_score": 0.0,\n'
            '  "reason": "Short explanation with object and direction (max 10 words)",\n'
            '  "navigation": "Clear directional guidance for the user"\n'
            "}\n\n"
            "Examples:\n"
            "- Clear path: {\"danger_score\": 0.0, \"reason\": \"Path clear\", \"navigation\": \"Move forward\"}\n"
            "- Obstacle: {\"danger_score\": 0.8, \"reason\": \"Chair directly ahead\", \"navigation\": \"Stop! Move left to avoid chair\"}"
        )

    def _build_user_prompt(self, frame_info: Dict[str, Any]) -> str:
        """
        Build a concise user prompt from categorized detections provided by main.py.
        Expects keys like 'left', 'right', 'front', 'ground' with lists of info strings.
        """
        try:
            def section(name: str) -> str:
                items = frame_info.get(name, [])
                if not items:
                    return f"{name.title()}: none"
                # Ensure one item per line for readability
                return f"{name.title()}:\n- " + "\n- ".join(str(x) for x in items)

            parts = [
                "Detections by region:",
                section("left"),
                section("front"),
                section("right"),
                section("ground"),
                "\nPlease analyze for immediate and potential obstacles with emphasis on the 'ground' region.",
            ]
            return "\n".join(parts)
        except Exception:
            logging.exception("Error building user prompt from frame_info; falling back to raw string")
            return str(frame_info)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object from model text output. Tries strict json first, then code-fence
        stripping, then best-effort literal_eval. Returns None if parsing fails.
        """
        if not text:
            return None
        candidate = text.strip()
        # Strip code fences if present
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", candidate)
        # Try to locate the first {...} block
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            candidate = match.group(0)
        # Strict JSON
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Safe python literal as fallback
        try:
            loaded = ast.literal_eval(candidate)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        logging.error("Failed to parse model JSON output")
        return None

    def _encode_image_base64(self, image_path: str) -> Optional[str]:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            import base64
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            logging.exception(f"Error encoding image {image_path} to base64")
            return None

    def _create_openai_chat_completion(self, client, model: str, messages: list, max_tokens: int, temperature: float = 0.7) -> Optional[str]:
        """
        Create an OpenAI chat completion with proper model handling for GPT-5 variants.
        
        Args:
            client: OpenAI client instance
            model: Model name
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            
        Returns:
            Response content or None if failed
        """
        try:
            # Handle GPT-5 variants that need actual model name from config
            if model in ["gpt-5", "gpt-5-nano"]:
                model_config = self.model_manager.get_model_config(model)
                actual_model_name = model_config.name if model_config else model
            else:
                actual_model_name = model
            
            response = client.chat.completions.create(
                model=actual_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip() if hasattr(response.choices[0].message, 'content') else None
        except Exception as e:
            logging.warning(f"Error creating OpenAI chat completion: {e}")
            return None


    def process_frame(self, frame_info: Dict[str, Any], model: ModelType = "gpt-4o") -> Optional[GPTResponse]:
        """Process a single frame using the specified model via model manager."""
        start_time = time.time()
        try:
            # Set the model in the model manager
            if not self.model_manager.set_current_model(model):
                logging.warning(f"Model {model} not available, using current model: {self.model_manager.get_current_model()}")
            
            # Get the appropriate response based on model
            response = self._get_model_response(frame_info, model)

            if response:
                logging.info(f"Raw {model} response: {response}")
                parsed = self._extract_json(response)
                if parsed is None:
                    logging.error(f"Failed to parse {model} JSON response: {response}")
                    return None
                logging.info(f"Parsed {model} response: {parsed}")
                return GPTResponse(
                    danger_score=float(parsed.get("danger_score", 0)),
                    reason=str(parsed.get("reason", "No data")),
                    navigation=str(parsed.get("navigation", "Path clear")),
                    processing_time=time.time() - start_time,
                    model_used=model
                )
        except Exception:
            logging.exception(f"Error processing frame with model {model}")
        return None
    
    def _get_model_response(self, frame_info: Dict[str, Any], model: ModelType) -> Optional[str]:
        """Get response from the specified model."""
        try:
            if model in ["gpt-4o", "gpt-5", "gpt-5-nano"]:
                return self._get_openai_response(frame_info, model)
            elif model == "claude-3.5":
                return self._get_anthropic_response(frame_info)
            elif model == "llama-3.2-vision":
                return self.get_llama_response(frame_info)
            else:
                logging.error(f"Unknown model: {model}")
                return None
        except Exception:
            logging.exception(f"Error getting response from {model}")
            return None

    def _get_openai_response(self, frame_info: Dict[str, Any], model: str) -> Optional[str]:
        """Get response from OpenAI models (GPT-4o, GPT-5, GPT-5-nano)."""
        try:
            # Check if model is available
            if not self.model_manager.is_model_available(model):
                logging.warning(f"Model {model} is not available, falling back to GPT-4o")
                if model != "gpt-4o":  # Prevent infinite recursion
                    return self._get_openai_response(frame_info, "gpt-4o")
                else:
                    return None
            
            user_prompt = self._build_user_prompt(frame_info) + "\n\n" + self.format_prompt
            client = self.model_manager.get_client(model)
            
            if not client:
                logging.error(f"No client available for {model}")
                return None
                
            # Use helper method for OpenAI API call
            messages = [
                {"role": "system", "content": self.system_prompt + "\n\n" + self.obstacle_rules},
                {"role": "user", "content": user_prompt}
            ]
            content = self._create_openai_chat_completion(
                client, model, messages,
                max_tokens=getattr(config, "GPT_MAX_TOKENS", 150),
                temperature=getattr(config, "GPT_TEMPERATURE", 0.7)
            )
            if content:
                logging.info(f"GPT-5 response content: '{content}'")
            return content
        except Exception as e:
            logging.warning(f"Error with {model}: {e}")
            # Only fallback if not already trying GPT-4o
            if model != "gpt-4o":
                logging.info("Falling back to GPT-4o")
                return self._get_openai_response(frame_info, "gpt-4o")
            else:
                logging.error("GPT-4o also failed, no fallback available")
                return None
    
    def _get_anthropic_response(self, frame_info: Dict[str, Any]) -> Optional[str]:
        """Get response from Anthropic Claude."""
        try:
            user_prompt = self._build_user_prompt(frame_info) + "\n\n" + self.format_prompt
            client = self.model_manager.get_client("claude-3.5")
            
            if not client:
                logging.error("No Anthropic client available")
                return None
                
            message = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=getattr(config, "GPT_MAX_TOKENS", 150),
                temperature=getattr(config, "GPT_TEMPERATURE", 0.7),
                system=self.system_prompt + "\n\n" + self.obstacle_rules,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return message.content[0].text
        except Exception:
            logging.exception("Error getting Claude response")
            return None

    def get_llama_response(self, frame_info: Dict[str, Any]) -> Optional[str]:
        """(Placeholder) Get LLaMA-3.2-Vision response."""
        try:
            logging.info("LLaMA-3.2-Vision API request not implemented yet")
            return None  # Implement API call if needed
        except Exception:
            logging.exception("Error getting LLaMA response")
            return None
    
    def query_gpt(self, prompt: str, max_tokens: int = 200, model: str = "gpt-4o") -> Optional[str]:
        """
        General purpose GPT query method for scene descriptions and other text tasks.
        
        Args:
            prompt: The prompt to send to GPT
            max_tokens: Maximum tokens in response
            model: Model to use (gpt-4o, gpt-5, gpt-5-nano, claude-3.5, etc.)
        
        Returns:
            The model's text response
        """
        try:
            # Set the model in the model manager
            if not self.model_manager.set_current_model(model):
                logging.warning(f"Model {model} not available, using current model: {self.model_manager.get_current_model()}")
                model = self.model_manager.get_current_model()
            
            if model in ["gpt-4o", "gpt-5", "gpt-5-nano"]:
                return self._query_openai(prompt, max_tokens, model)
            elif model == "claude-3.5":
                return self._query_anthropic(prompt, max_tokens)
            else:
                logging.warning(f"Unsupported model: {model}, falling back to gpt-4o")
                return self._query_openai(prompt, max_tokens, "gpt-4o")
        except Exception:
            logging.exception(f"Error querying {model}")
            return None

    def analyze_image(self, image_path: str, prompt: str, max_tokens: int = 500, model: str = "gpt-4o") -> Optional[str]:
        """
        Analyze an image using vision models for pilot mode.
        
        Args:
            image_path: Path to the image file
            prompt: The prompt for image analysis
            max_tokens: Maximum tokens in response
            model: Model to use (gpt-4o, gpt-5, gpt-5-nano, claude-3.5, llama-3.2-vision)
        
        Returns:
            The model's text response
        """
        try:
            # Set the model in the model manager
            if not self.model_manager.set_current_model(model):
                logging.warning(f"Model {model} not available, using current model: {self.model_manager.get_current_model()}")
                model = self.model_manager.get_current_model()
            
            if model in ["gpt-4o", "gpt-5", "gpt-5-nano"]:
                return self._analyze_image_openai(image_path, prompt, max_tokens, model)
            elif model == "claude-3.5":
                return self._analyze_image_anthropic(image_path, prompt, max_tokens)
            elif model == "llama-3.2-vision":
                return self._analyze_image_llama(image_path, prompt, max_tokens)
            else:
                logging.warning(f"Unsupported vision model: {model}, falling back to gpt-4o")
                return self._analyze_image_openai(image_path, prompt, max_tokens, "gpt-4o")
        except Exception:
            logging.exception(f"Error analyzing image with {model}")
            return None

    def _analyze_image_openai(self, image_path: str, prompt: str, max_tokens: int, model: str) -> Optional[str]:
        """Analyze image using OpenAI vision models."""
        try:
            # Validate image file exists
            if not os.path.exists(image_path):
                logging.error(f"Image file does not exist: {image_path}")
                return None
            
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                logging.error(f"Image file is empty: {image_path}")
                return None
            
            logging.debug(f"Analyzing image: {image_path} (size: {file_size} bytes)")
            
            # Encode image to base64
            base64_image = self._encode_image_base64(image_path)
            if not base64_image:
                logging.error(f"Failed to encode image to base64: {image_path}")
                return None
            
            logging.debug(f"Image encoded to base64 (length: {len(base64_image)} characters)")
            
            # Get OpenAI client - use model name (e.g., "gpt-4o"), not provider name
            openai_client = self.model_manager.get_client(model)
            if not openai_client:
                # Try fallback to gpt-4o if different model was requested
                openai_client = self.model_manager.get_client("gpt-4o")
                if not openai_client:
                    logging.error(f"OpenAI client not available for model: {model}")
                    return None
                logging.info(f"Using gpt-4o client for {model}")
            
            # Prepare messages for vision model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            logging.debug(f"Calling OpenAI vision API with model: {model}, max_tokens: {max_tokens}")
            
            # Make API call using helper method
            response = self._create_openai_chat_completion(
                openai_client, model, messages, max_tokens, temperature=0.7
            )
            
            if response:
                logging.info(f"OpenAI vision API returned response (length: {len(response)} characters)")
            else:
                logging.warning("OpenAI vision API returned None or empty response")
            
            return response
            
        except FileNotFoundError as e:
            logging.error(f"Image file not found: {e}")
            return None
        except Exception as e:
            logging.exception(f"Error in OpenAI image analysis: {e}")
            return None

    def _analyze_image_anthropic(self, image_path: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Analyze image using Anthropic Claude vision model."""
        try:
            # Encode image to base64
            base64_image = self._encode_image_base64(image_path)
            if not base64_image:
                return None
            
            # Get Anthropic client
            anthropic_client = self.model_manager.get_client("anthropic")
            if not anthropic_client:
                logging.error("Anthropic client not available")
                return None
            
            # Make API call
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception:
            logging.exception("Error in Anthropic image analysis")
            return None

    def _analyze_image_llama(self, image_path: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Analyze image using LLaMA vision model (placeholder)."""
        try:
            # For now, fall back to text-only analysis
            logging.warning("LLaMA vision model not fully implemented, using text-only analysis")
            return self.query_gpt(prompt, max_tokens, "gpt-4o")
        except Exception:
            logging.exception("Error in LLaMA image analysis")
            return None
    
    @func_set_timeout(15)
    def _query_openai(self, prompt: str, max_tokens: int, model: str = "gpt-4o") -> Optional[str]:
        """Query OpenAI GPT with timeout protection."""
        try:
            # Check if model is available
            if not self.model_manager.is_model_available(model):
                logging.warning(f"Model {model} is not available, falling back to GPT-4o")
                if model != "gpt-4o":  # Prevent infinite recursion
                    return self._query_openai(prompt, max_tokens, "gpt-4o")
                else:
                    return None
                    
            client = self.model_manager.get_client(model)
            if not client:
                logging.error(f"No OpenAI client available for {model}")
                return None
                
            # Use helper method for OpenAI API call
            messages = [
                {"role": "system", "content": "You are a helpful assistant that provides clear, concise descriptions."},
                {"role": "user", "content": prompt}
            ]
            return self._create_openai_chat_completion(client, model, messages, max_tokens, temperature=0.7)
        except Exception as e:
            logging.warning(f"Error with {model}: {e}")
            # Only fallback if not already trying GPT-4o
            if model != "gpt-4o":
                logging.info("Falling back to GPT-4o")
                return self._query_openai(prompt, max_tokens, "gpt-4o")
            else:
                logging.error("GPT-4o also failed, no fallback available")
                return None
    
    @func_set_timeout(15)
    def _query_anthropic(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Query Anthropic Claude with timeout protection."""
        try:
            client = self.model_manager.get_client("claude-3.5")
            if not client:
                logging.error("No Anthropic client available")
                return None
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.7,
                system="You are a helpful assistant that provides clear, concise descriptions.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception:
            logging.exception("Error in Anthropic query")
            return None
    
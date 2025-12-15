"""
Model Manager Module for AIDGPT
Handles all LLM model switching, initialization, and API management.
"""

import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union
from dataclasses import dataclass
from openai import OpenAI
from anthropic import Anthropic

# Model type definitions
ModelType = Literal["gpt-4o", "gpt-5", "gpt-5-nano", "claude-3.5", "llama-3.2-vision"]

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    api_key_env: str
    api_key: str = ""
    client: Optional[Any] = None
    max_tokens: int = 200
    temperature: float = 0.7
    timeout: int = 10
    available: bool = False

class ModelManager:
    """Centralized model management for all LLM operations."""
    
    def __init__(self, api_key_path: str = "API_KEYS.yaml"):
        """
        Initialize the model manager with API keys.
        
        Args:
            api_key_path: Path to the YAML file containing API keys
        """
        # Resolve path relative to this script's directory if path is relative
        if not os.path.isabs(api_key_path):
            # Get the directory where this script is located
            script_dir = Path(__file__).parent.absolute()
            self.api_key_path = str(script_dir / api_key_path)
        else:
            self.api_key_path = api_key_path
        self.models: Dict[str, ModelConfig] = {}
        self.current_model: str = "gpt-4o"
        self.fallback_model: str = "gpt-4o"
        
        # Initialize all models
        self._load_api_keys()
        self._initialize_models()
        
        logging.info(f"ModelManager initialized with {len(self.models)} models")
        logging.info(f"Current model: {self.current_model}")
        logging.info(f"Available models: {[name for name, config in self.models.items() if config.available]}")
    
    def _load_api_keys(self) -> None:
        """Load API keys from YAML file."""
        try:
            with open(self.api_key_path, "r") as f:
                data = yaml.safe_load(f)
                
            # Store API keys for each provider
            self.api_keys = {
                "openai": data.get("OPENAI_API_KEY", ""),
                "anthropic": data.get("ANTHROPIC_API_KEY", ""),
                "llama": data.get("LLAMA_API_KEY", ""),
                "groq": data.get("GROQ_API_KEY", "")
            }
            
            # Log which keys were loaded
            for provider, key in self.api_keys.items():
                if key:
                    logging.info(f"{provider.upper()} API key loaded successfully")
                else:
                    logging.warning(f"{provider.upper()} API key not found")
                    
        except Exception as e:
            logging.exception(f"Error loading API keys from {self.api_key_path}")
            raise
    
    def _initialize_models(self) -> None:
        """Initialize all available models."""
        
        # GPT-4o Configuration
        self.models["gpt-4o"] = ModelConfig(
            name="gpt-4o",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            api_key=self.api_keys["openai"],
            max_tokens=200,
            temperature=0.7,
            timeout=10
        )
        
        # GPT-5 Configuration (using working model)
        self.models["gpt-5"] = ModelConfig(
            name="gpt-5-chat-latest",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            api_key=self.api_keys["openai"],
            max_tokens=200,
            temperature=0.7,
            timeout=10
        )
        
        # GPT-5-nano Configuration (using working model)
        self.models["gpt-5-nano"] = ModelConfig(
            name="gpt-5-chat-latest",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
            api_key=self.api_keys["openai"],
            max_tokens=200,
            temperature=0.7,
            timeout=10
        )
        
        # Claude-3.5 Configuration
        self.models["claude-3.5"] = ModelConfig(
            name="claude-3.5",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            api_key=self.api_keys["anthropic"],
            max_tokens=200,
            temperature=0.7,
            timeout=10
        )
        
        # LLaMA-3.2-Vision Configuration
        self.models["llama-3.2-vision"] = ModelConfig(
            name="llama-3.2-vision",
            provider="llama",
            api_key_env="LLAMA_API_KEY",
            api_key=self.api_keys["llama"],
            max_tokens=200,
            temperature=0.7,
            timeout=10
        )
        
        # Initialize clients for available models
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize API clients for available models."""
        
        # Initialize OpenAI client for GPT models
        if self.api_keys["openai"]:
            try:
                openai_client = OpenAI(api_key=self.api_keys["openai"])
                
                # Set client for all OpenAI models
                for model_name in ["gpt-4o", "gpt-5", "gpt-5-nano"]:
                    self.models[model_name].client = openai_client
                    self.models[model_name].available = True
                    
                logging.info("OpenAI client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize Anthropic client for Claude
        if self.api_keys["anthropic"]:
            try:
                anthropic_client = Anthropic(api_key=self.api_keys["anthropic"])
                self.models["claude-3.5"].client = anthropic_client
                self.models["claude-3.5"].available = True
                logging.info("Anthropic client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Initialize LLaMA client (placeholder)
        if self.api_keys["llama"]:
            try:
                # Placeholder for LLaMA client initialization
                # self.models["llama-3.2-vision"].client = llama_client
                self.models["llama-3.2-vision"].available = True
                logging.info("LLaMA client placeholder initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize LLaMA client: {e}")
    
    def set_current_model(self, model_name: str) -> bool:
        """
        Set the current active model.
        
        Args:
            model_name: Name of the model to set as current
            
        Returns:
            True if model was set successfully, False otherwise
        """
        if model_name not in self.models:
            logging.error(f"Unknown model: {model_name}")
            return False
            
        if not self.models[model_name].available:
            logging.warning(f"Model {model_name} is not available, falling back to {self.fallback_model}")
            self.current_model = self.fallback_model
            return False
            
        self.current_model = model_name
        logging.info(f"Current model set to: {model_name}")
        return True
    
    def get_current_model(self) -> str:
        """Get the current active model name."""
        return self.current_model
    
    def get_model_config(self, model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model (uses current model if None)
            
        Returns:
            ModelConfig object or None if not found
        """
        if model_name is None:
            model_name = self.current_model
            
        return self.models.get(model_name)
    
    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return [name for name, config in self.models.items() if config.available]
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return model_name in self.models and self.models[model_name].available
    
    def get_client(self, model_name: Optional[str] = None) -> Optional[Any]:
        """
        Get the API client for a specific model.
        
        Args:
            model_name: Name of the model (uses current model if None)
            
        Returns:
            API client or None if not available
        """
        config = self.get_model_config(model_name)
        return config.client if config else None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all models."""
        info = {
            "current_model": self.current_model,
            "fallback_model": self.fallback_model,
            "total_models": len(self.models),
            "available_models": self.get_available_models(),
            "models": {}
        }
        
        for name, config in self.models.items():
            info["models"][name] = {
                "provider": config.provider,
                "available": config.available,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "timeout": config.timeout,
                "has_client": config.client is not None
            }
            
        return info
    
    def validate_model_switch(self, new_model: str) -> tuple[bool, str]:
        """
        Validate if a model switch is possible.
        
        Args:
            new_model: Name of the model to switch to
            
        Returns:
            Tuple of (is_valid, message)
        """
        if new_model not in self.models:
            return False, f"Unknown model: {new_model}"
            
        if not self.models[new_model].available:
            return False, f"Model {new_model} is not available (missing API key or client)"
            
        return True, f"Model {new_model} is ready to use"
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a new model with validation.
        
        Args:
            new_model: Name of the model to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        is_valid, message = self.validate_model_switch(new_model)
        
        if not is_valid:
            logging.warning(message)
            return False
            
        success = self.set_current_model(new_model)
        if success:
            logging.info(f"Successfully switched to {new_model}")
        else:
            logging.error(f"Failed to switch to {new_model}")
            
        return success
    
    def get_fallback_model(self) -> str:
        """Get the fallback model name."""
        return self.fallback_model
    
    def set_fallback_model(self, model_name: str) -> bool:
        """
        Set the fallback model.
        
        Args:
            model_name: Name of the fallback model
            
        Returns:
            True if fallback model was set successfully
        """
        if model_name in self.models and self.models[model_name].available:
            self.fallback_model = model_name
            logging.info(f"Fallback model set to: {model_name}")
            return True
        else:
            logging.error(f"Cannot set fallback model to {model_name}: not available")
            return False

# Global model manager instance
_model_manager: Optional[ModelManager] = None

def get_model_manager(api_key_path: str = "API_KEYS.yaml") -> ModelManager:
    """
    Get the global model manager instance.
    
    Args:
        api_key_path: Path to API keys file (only used on first call)
        
    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(api_key_path)
    return _model_manager

def initialize_models(api_key_path: str = "API_KEYS.yaml") -> ModelManager:
    """
    Initialize the model manager (alias for get_model_manager).
    
    Args:
        api_key_path: Path to API keys file
        
    Returns:
        ModelManager instance
    """
    return get_model_manager(api_key_path)

# Convenience functions for common operations
def get_current_model() -> str:
    """Get the current active model name."""
    return get_model_manager().get_current_model()

def switch_to_model(model_name: str) -> bool:
    """Switch to a specific model."""
    return get_model_manager().switch_model(model_name)

def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    return get_model_manager().is_model_available(model_name)

def get_available_models() -> list[str]:
    """Get list of available models."""
    return get_model_manager().get_available_models()

def get_model_client(model_name: Optional[str] = None):
    """Get the API client for a model."""
    return get_model_manager().get_client(model_name)

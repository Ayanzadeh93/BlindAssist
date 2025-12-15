#!/usr/bin/env python3
"""Frame processing module for AIDGPT application."""
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from application_state import ApplicationState
from app_utils import validate_frame, create_placeholder_frame


def categorize_detections(detections: List, frame_id: int,
                         frame_width: int, frame_height: int) -> Dict:
    """Categorize detections based on their position in the frame."""
    categorized = {'frame_id': frame_id, 'left': [], 'right': [], 'front': [], 'ground': []}
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        class_name = detection.class_name
        confidence = detection.confidence
        height_pct = (y2 - y1) / frame_height * 100
        width_pct = (x2 - x1) / frame_width * 100
        center_x = (x1 + x2) / 2 / frame_width * 100
        center_y = (y1 + y2) / 2 / frame_height * 100
        info = (
            f'class:{class_name}, confidence:{confidence:.2f}, '
            f'center_x:{center_x:.1f}%, center_y:{center_y:.1f}%, '
            f'height:{height_pct:.1f}%, width:{width_pct:.1f}%'
        )
        if center_x < 25:
            categorized['left'].append(info)
        elif center_x > 75:
            categorized['right'].append(info)
        elif center_y < 50:
            categorized['front'].append(info)
        else:
            categorized['ground'].append(info)
    return categorized


class DepthEstimator:
    """Depth estimation using Depth Anything V2 Tiny."""

    def __init__(self):
        self.depth_processor = None
        self.depth_model = None
        self.depth_device = None
        self.initialized = False
        self.initialization_attempted = False
        self.use_colormap = True
        # Cache for sharing depth computation
        self._cached_raw_depth: Optional[np.ndarray] = None
        self._cached_frame_hash: int = 0

    def _initialize_depth_model(self) -> None:
        if self.initialization_attempted:
            return
        self.initialization_attempted = True
        if not DEPTH_ANYTHING_AVAILABLE:
            logging.warning("Depth Anything V2 not available. Install with: pip install transformers accelerate")
            return
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch not available. Install with: pip install torch torchvision")
            return
        try:
            logging.info("Loading Depth Anything V2 Tiny model (lazy)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_names = [
                "LiheYoung/depth-anything-small-hf",
                "depth-anything/DA3-SMALL",
                "LiheYoung/depth-anything-v2-small-hf",
            ]
            for model_name in model_names:
                try:
                    logging.info(f"Attempting to load model: {model_name}")
                    self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
                    self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
                    logging.info(f"✓ Successfully loaded model: {model_name}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {str(e)[:100]}")
                    continue
            if self.depth_model is None:
                raise RuntimeError("Failed to load any Depth Anything V2 model.")
            self.depth_model = self.depth_model.to(device)
            self.depth_model.eval()
            self.depth_device = device
            self.initialized = True
            logging.info(f"✓ Depth Anything V2 initialized on {device}")
        except Exception as e:
            logging.error(f"Failed to initialize Depth Anything V2: {e}")
            self.depth_processor = None
            self.depth_model = None
            self.initialized = False

    def _compute_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Internal method to compute raw depth (0-1 normalized).
        
        Optimized with better caching and reduced computation.
        
        Returns:
            Normalized depth map (0-1 float32) or None on error
        """
        if frame is None or frame.size == 0:
            return None
        
        # Optimized cache check: use frame shape and a small sample for hash
        # This is faster than hashing entire frame
        h, w = frame.shape[:2]
        frame_sample = frame[::max(1, h//20), ::max(1, w//20)]  # Sample every 20th pixel
        frame_hash = hash((h, w, frame_sample.tobytes()[:500]))  # Hash shape + sample
        
        if frame_hash == self._cached_frame_hash and self._cached_raw_depth is not None:
            return self._cached_raw_depth
        
        try:
            # Resize frame for faster processing if too large (maintain aspect ratio)
            max_dim = 512  # Process at max 512px for speed
            orig_h, orig_w = frame.shape[:2]
            if max(orig_h, orig_w) > max_dim:
                scale = max_dim / max(orig_h, orig_w)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_resized = frame
                new_h, new_w = orig_h, orig_w
            
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.depth_device)
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                if hasattr(outputs, 'predicted_depth'):
                    predicted_depth = outputs.predicted_depth
                elif hasattr(outputs, 'depth'):
                    predicted_depth = outputs.depth
                elif isinstance(outputs, torch.Tensor):
                    predicted_depth = outputs
                else:
                    predicted_depth = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]
            
            if len(predicted_depth.shape) == 4:
                predicted_depth = predicted_depth.squeeze(0)
            if len(predicted_depth.shape) == 3:
                predicted_depth = predicted_depth.squeeze(0)
            
            depth_h, depth_w = predicted_depth.shape[-2:]
            
            # Resize back to original if needed
            if depth_h != orig_h or depth_w != orig_w:
                depth_map = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(0).unsqueeze(0) if len(predicted_depth.shape) == 2 else predicted_depth.unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",  # Use bilinear instead of bicubic for speed
                    align_corners=False
                )
                depth_map = depth_map.squeeze().cpu().numpy()
            else:
                depth_map = predicted_depth.cpu().numpy()
            
            # Normalize
            depth_min = float(depth_map.min())
            depth_max = float(depth_map.max())
            if depth_max > depth_min + 1e-6:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
            
            self._cached_raw_depth = depth_normalized.astype(np.float32)
            self._cached_frame_hash = frame_hash
            return self._cached_raw_depth
        except Exception:
            logging.exception("Error computing depth")
            return None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.initialization_attempted:
            self._initialize_depth_model()
        if not self.initialized or self.depth_model is None or self.depth_processor is None:
            h, w = frame.shape[:2] if frame is not None else (480, 640)
            placeholder = create_placeholder_frame(h, w, "Loading depth model...")
            cv2.putText(placeholder, "Please wait...", (10, h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return placeholder
        try:
            if not validate_frame(frame):
                h, w = 480, 640
                placeholder = np.zeros((h, w), dtype=np.uint8)
                if self.use_colormap:
                    placeholder = cv2.applyColorMap(placeholder, cv2.COLORMAP_VIRIDIS)
                return placeholder
            depth_normalized = self._compute_depth(frame)
            if depth_normalized is None:
                h, w = frame.shape[:2]
                placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                return placeholder
            # Convert to 0-255 uint8 for visualization
            depth_map = (depth_normalized * 255).astype(np.uint8)
            if self.use_colormap:
                depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
            return depth_map
        except Exception as e:
            logging.exception(f"Error estimating depth: {e}")
            h, w = frame.shape[:2] if frame is not None else (480, 640)
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Depth estimation error", (10, h//2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(placeholder, "Check logs for details", (10, h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return placeholder

    def is_available(self) -> bool:
        if not self.initialization_attempted:
            self._initialize_depth_model()
        return self.initialized

    def get_raw_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get raw normalized depth values (0-1 float32) for distance calculations.
        
        Uses cached computation if available.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Normalized depth map (0-1 float32) where higher = closer, or None on error
        """
        if not self.initialization_attempted:
            self._initialize_depth_model()
        if not self.initialized or self.depth_model is None or self.depth_processor is None:
            return None
        return self._compute_depth(frame)




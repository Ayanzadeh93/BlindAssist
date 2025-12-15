#!/usr/bin/env python3
"""Utility functions for common patterns in AIDGPT application."""
import numpy as np
from typing import Optional, Any, Tuple
from config import config


def get_config_value(args: Any, attr_name: Optional[str], config_attr: str, default: Any) -> Any:
    """Get configuration value from args or config with fallback.
    
    Args:
        args: Arguments namespace (can be None)
        attr_name: Attribute name in args (can be None)
        config_attr: Attribute name in config
        default: Default value if neither exists
        
    Returns:
        Configuration value
    """
    if args is not None and attr_name is not None:
        value = getattr(args, attr_name, None)
        if value is not None:
            return value
    return getattr(config, config_attr, default)


def validate_frame(frame: Optional[np.ndarray]) -> bool:
    """Validate that a frame is usable.
    
    Args:
        frame: Frame to validate
        
    Returns:
        True if frame is valid, False otherwise
    """
    return frame is not None and hasattr(frame, 'size') and frame.size > 0


def create_placeholder_frame(height: int = 480, width: int = 640, 
                           message: Optional[str] = None,
                           color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Create a placeholder frame with optional message.
    
    Args:
        height: Frame height
        width: Frame width
        message: Optional text message to display
        color: BGR color for text
        
    Returns:
        Placeholder frame as uint8 array
    """
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    if message:
        import cv2
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = (width - text_size[0]) // 2
        y = height // 2
        cv2.putText(placeholder, message, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return placeholder

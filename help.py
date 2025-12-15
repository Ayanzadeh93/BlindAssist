#!/usr/bin/env python3
"""Help system module for displaying help overlay and handling help-related interactions."""
import cv2
import numpy as np
from typing import Optional


class HelpSystem:
    """Manages help overlay display and interaction."""
    
    def __init__(self) -> None:
        """Initialize the help system."""
        self.show_help: bool = False
        self.scroll_offset: int = 0
        self.line_height: int = 25
        self.max_scroll: int = 0
        
    def toggle_help(self) -> None:
        """Toggle help display on/off."""
        self.show_help = not self.show_help
        if not self.show_help:
            self.scroll_offset = 0
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input for help system.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            True if key was handled by help system, False otherwise
        """
        if not self.show_help:
            return False
            
        # ESC key (27) or 'h' to close help
        if key == 27 or key == ord('h'):
            self.toggle_help()
            return True
            
        # Arrow keys for scrolling
        if key == 82 or key == 0:  # Up arrow
            self.scroll_offset = max(0, self.scroll_offset - 1)
            return True
        elif key == 84 or key == 1:  # Down arrow
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
            return True
            
        return False
    
    def handle_mouse_scroll(self, delta: int) -> None:
        """Handle mouse scroll events for help system.
        
        Args:
            delta: Scroll delta (positive for down, negative for up)
        """
        if not self.show_help:
            return
            
        if delta > 0:
            self.scroll_offset = min(self.max_scroll, self.scroll_offset + 1)
        else:
            self.scroll_offset = max(0, self.scroll_offset - 1)
    
    def add_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add help overlay to the frame.
        
        Args:
            frame: Input frame to add overlay to
            
        Returns:
            Frame with help overlay added
        """
        if frame is None:
            return frame
            
        # Create a semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Help content
        help_lines = [
            "=== AIDGPT Help ===",
            "",
            "Keyboard Controls:",
            "  Q - Quit application",
            "  P - Pause/Resume",
            "  D - Toggle debug overlay",
            "  M - Toggle depth map",
            "  B - Toggle detection boxes",
            "  G - Toggle GPT overlay",
            "  A - Toggle audio",
            "  T - Test audio",
            "  V - Toggle detection alerts",
            "  S - Toggle smart mode",
            "  K - Toggle spatial understanding",
            "  J - Deep spatial (depth + detections)",
            "  R - Toggle ArUco marker detection (000-999)",
            "  H - Toggle help (this window)",
            "",
            "Mouse Controls:",
            "  Left Click - Toggle debug overlay",
            "  Right Click - Pause/Resume",
            "  Scroll - Navigate help (when open)",
            "",
            "Navigation:",
            "  Arrow Keys / Mouse Scroll - Scroll help",
            "  ESC or H - Close help",
            "",
            "Press H or ESC to close this help window."
        ]
        
        # Calculate text positioning
        start_y = 30 - (self.scroll_offset * self.line_height)
        x_offset = 30
        
        # Draw help text
        for i, line in enumerate(help_lines):
            y_pos = start_y + (i * self.line_height)
            
            # Skip if outside visible area
            if y_pos < 0 or y_pos > h - 20:
                continue
                
            # Style header differently
            if line.startswith("==="):
                font_scale = 0.8
                thickness = 2
                color = (0, 255, 255)  # Cyan
            elif line and not line.startswith("  ") and line.endswith(":"):
                font_scale = 0.6
                thickness = 2
                color = (0, 255, 0)  # Green
            else:
                font_scale = 0.5
                thickness = 1
                color = (255, 255, 255)  # White
                
            cv2.putText(
                frame,
                line,
                (x_offset, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
        
        # Calculate max scroll
        total_height = len(help_lines) * self.line_height
        visible_height = h - 60
        self.max_scroll = max(0, (total_height - visible_height) // self.line_height)
        
        return frame










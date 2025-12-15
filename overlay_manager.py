#!/usr/bin/env python3
"""Overlay Manager for AIDGPT."""
import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum
import logging
from config import config


class OverlayType(Enum):
    DANGER = "danger"
    NAVIGATION = "navigation"
    STATUS = "status"
    INFO = "info"
    SMART = "smart"


@dataclass
class OverlayMessage:
    text: str
    overlay_type: OverlayType = OverlayType.INFO
    danger_score: float = 0.0
    navigation: str = ""
    timestamp: float = field(default_factory=time.monotonic)
    duration: float = 2.0
    fade_start: float = 0.5

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) >= self.duration

    @property
    def remaining_time(self) -> float:
        elapsed = time.monotonic() - self.timestamp
        return max(0.0, self.duration - elapsed)

    @property
    def opacity(self) -> float:
        remaining = self.remaining_time
        if remaining <= 0:
            return 0.0
        if remaining <= self.fade_start:
            return remaining / self.fade_start
        return 1.0


class OverlayManager:
    def __init__(self, overlay_duration: float = 2.0):
        self.overlay_duration = overlay_duration
        self._overlays: Dict[OverlayType, OverlayMessage] = {}
        self._lock = threading.Lock()
        self._is_processing = False
        self._processing_start_time = 0.0
        logging.info(f"OverlayManager initialized with {overlay_duration}s duration")

    def set_overlay(self, message: OverlayMessage) -> None:
        with self._lock:
            self._overlays[message.overlay_type] = message

    def set_danger_overlay(self, text: str, danger_score: float, navigation: str = "") -> None:
        message = OverlayMessage(
            text=text,
            overlay_type=OverlayType.DANGER,
            danger_score=danger_score,
            navigation=navigation,
            duration=self.overlay_duration
        )
        self.set_overlay(message)

    def set_navigation_overlay(self, text: str) -> None:
        message = OverlayMessage(
            text=text,
            overlay_type=OverlayType.NAVIGATION,
            duration=self.overlay_duration
        )
        self.set_overlay(message)

    def set_processing_state(self, is_processing: bool) -> None:
        with self._lock:
            self._is_processing = is_processing
            if is_processing:
                self._processing_start_time = time.monotonic()
    
    def is_processing(self) -> bool:
        """Check if GPT/inference is currently processing."""
        with self._lock:
            return self._is_processing

    def clear_all_overlays(self) -> None:
        with self._lock:
            self._overlays.clear()

    def _cleanup_expired(self) -> None:
        expired = [ot for ot, msg in self._overlays.items() if msg.is_expired]
        for ot in expired:
            del self._overlays[ot]

    def get_danger_color(self, danger_score: float, opacity: float = 1.0) -> Tuple[int, int, int]:
        if danger_score >= 0.7:
            base_color = (0, 0, 255)
        elif danger_score >= 0.4:
            base_color = (0, 165, 255)
        else:
            base_color = (0, 255, 0)
        return tuple(int(c * opacity) for c in base_color)

    def render_overlays(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return frame
        output = frame.copy()
        with self._lock:
            self._cleanup_expired()
            if self._is_processing and not self._overlays:
                output = self._render_processing_overlay(output)
            for overlay_type in [OverlayType.INFO, OverlayType.STATUS,
                                 OverlayType.NAVIGATION, OverlayType.SMART,
                                 OverlayType.DANGER]:
                if overlay_type in self._overlays:
                    message = self._overlays[overlay_type]
                    output = self._render_overlay(output, message)
        return output

    def _render_processing_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Import UI config with fallback
        try:
            ui = config.UI_COLORS
            layout = config.UI_LAYOUT
            fonts = config.UI_FONTS
        except AttributeError:
            ui = {'bg_dark': (30, 30, 30), 'processing': (0, 200, 255)}
            layout = {'padding': 15}
            fonts = {'title_scale': 0.8, 'title_thickness': 2}
        
        bar_height = 70
        bar_y = h - bar_height - 10
        
        # Direct rectangle draw - no frame copying
        cv2.rectangle(frame, (10, bar_y), (w - 10, h - 10), ui.get('bg_dark', (30, 30, 30)), -1)
        
        elapsed = time.monotonic() - self._processing_start_time
        
        # Animated dots
        dots_count = int(elapsed * 2) % 4
        dots = "." * dots_count
        text = f"AI Processing{dots}"
        
        # Draw text - removed LINE_AA for performance
        cv2.putText(frame, text, (25, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, fonts.get('title_scale', 0.8),
                   ui.get('processing', (0, 200, 255)), fonts.get('title_thickness', 2))
        
        # Draw animated progress bar
        bar_width = w - 50
        bar_x = 25
        bar_fill_y = h - 20
        bar_bg_height = 4
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_fill_y), (bar_x + bar_width, bar_fill_y + bar_bg_height),
                     (60, 60, 60), -1)
        
        # Animated fill
        progress = (elapsed % 2.0) / 2.0  # 0 to 1 cycle every 2 seconds
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_fill_y), (bar_x + fill_width, bar_fill_y + bar_bg_height),
                         ui.get('processing', (0, 200, 255)), -1)
        
        return frame

    def _render_overlay(self, frame: np.ndarray, message: OverlayMessage) -> np.ndarray:
        h, w = frame.shape[:2]
        opacity = message.opacity
        if opacity <= 0:
            return frame
        
        # Import UI config
        try:
            ui = config.UI_COLORS
            layout = config.UI_LAYOUT
            fonts = config.UI_FONTS
        except AttributeError:
            # Fallback if UI config not available
            ui = {'bg_dark': (30, 30, 30), 'text_primary': (255, 255, 255), 
                  'danger': (60, 60, 255), 'warning': (0, 165, 255), 'success': (80, 200, 120)}
            layout = {'padding': 15, 'corner_radius': 8}
            fonts = {'title_scale': 0.8, 'title_thickness': 2, 'body_scale': 0.5, 'body_thickness': 1}
        
        bg_opacity = 0.7 * opacity

        if message.overlay_type in [OverlayType.DANGER, OverlayType.SMART]:
            bar_height = 95 if message.navigation else 65
            bar_y = h - bar_height - 10
            
            # Single draw operation - no overlay blending for performance
            cv2.rectangle(frame, (10, bar_y), (w - 10, h - 10), ui.get('bg_dark', (30, 30, 30)), -1)
            
            # Danger score with color gradient
            color = self.get_danger_color(message.danger_score, 1.0)
            
            # Danger score bar (visual indicator)
            bar_width = w - 40
            bar_x = 20
            bar_fill_y = bar_y + 15
            bar_bg_height = 8
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_fill_y), (bar_x + bar_width, bar_fill_y + bar_bg_height), 
                         (60, 60, 60), -1)
            
            # Fill bar based on danger score
            fill_width = int(bar_width * message.danger_score)
            if fill_width > 0:
                cv2.rectangle(frame, (bar_x, bar_fill_y), (bar_x + fill_width, bar_fill_y + bar_bg_height),
                             color, -1)
            
            # Danger text - removed LINE_AA for performance
            danger_text = f"DANGER {message.danger_score:.1f}"
            cv2.putText(frame, danger_text,
                       (bar_x, bar_fill_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts.get('body_scale', 0.5),
                       color, fonts.get('body_thickness', 1))
            
            # Reason text - removed LINE_AA
            reason_y = bar_fill_y + bar_bg_height + 25
            cv2.putText(frame, message.text,
                       (bar_x, reason_y),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts.get('title_scale', 0.8),
                       color, fonts.get('title_thickness', 2))
            
            # Navigation text - removed LINE_AA
            if message.navigation:
                nav_y = reason_y + 28
                cv2.putText(frame, f"> {message.navigation}",
                           (bar_x, nav_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           ui.get('text_primary', (255, 255, 255)), 2)
                
        elif message.overlay_type == OverlayType.NAVIGATION:
            bar_height = 55
            bar_y = h - bar_height - 10
            
            cv2.rectangle(frame, (10, bar_y), (w - 10, h - 10), ui.get('bg_dark', (30, 30, 30)), -1)
            
            color = tuple(int(c * opacity) for c in ui.get('success', (0, 255, 0)))
            cv2.putText(frame, f"> {message.text}", (25, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts.get('title_scale', 0.8),
                       color, fonts.get('title_thickness', 2))
                       
        elif message.overlay_type == OverlayType.STATUS:
            text_size = cv2.getTextSize(message.text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       fonts.get('body_scale', 0.5), fonts.get('body_thickness', 1))[0]
            x = w - text_size[0] - 30
            y = 35
            
            cv2.rectangle(frame, (x - 10, y - 20), (w - 10, y + 8), 
                         ui.get('bg_dark', (30, 30, 30)), -1)
            
            color = tuple(int(c * opacity) for c in ui.get('text_primary', (255, 255, 255)))
            cv2.putText(frame, message.text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts.get('body_scale', 0.5),
                       color, fonts.get('body_thickness', 1))
        else:  # INFO
            cv2.rectangle(frame, (10, 10), (400, 50), ui.get('bg_dark', (30, 30, 30)), -1)
            
            color = tuple(int(c * opacity) for c in ui.get('text_primary', (255, 255, 255)))
            cv2.putText(frame, message.text, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       color, 2)
        return frame

    def update_from_inference_result(self, result) -> None:
        if result is None:
            return
        if result.is_safety_critical or result.danger_score >= 0.4:
            self.set_danger_overlay(result.response_text, result.danger_score, result.navigation)
        elif result.navigation:
            self.set_navigation_overlay(result.navigation)
        else:
            self.set_overlay(OverlayMessage(
                text=result.response_text,
                overlay_type=OverlayType.INFO,
                duration=self.overlay_duration
            ))

    def has_active_overlay(self) -> bool:
        with self._lock:
            self._cleanup_expired()
            return len(self._overlays) > 0


_overlay_manager: Optional[OverlayManager] = None


def get_overlay_manager(overlay_duration: float = 2.0) -> OverlayManager:
    global _overlay_manager
    if _overlay_manager is None:
        _overlay_manager = OverlayManager(overlay_duration)
    return _overlay_manager


def reset_overlay_manager() -> None:
    global _overlay_manager
    _overlay_manager = None




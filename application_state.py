#!/usr/bin/env python3
"""Application state management module for AIDGPT.

Tracks runtime state including time-based inference schedule, overlay timing,
audio settings, and spatial mode flags.
"""
import time
from typing import Optional, Dict, Any, List
import numpy as np
from metrics import PerformanceMetrics


class ApplicationState:
    """Manages the application's runtime state and configuration."""

    def __init__(self) -> None:
        # Core flags
        self.running: bool = True
        self.paused: bool = False

        # Feature toggles
        self.show_debug: bool = False
        self.show_depth_map: bool = False
        self.show_detections: bool = True
        self.show_gpt_overlay: bool = True
        self.audio_enabled: bool = True
        self.detection_alert: bool = True
        self.smart_mode: bool = False
        self.show_ui_panel: bool = False  # controls/last-response overlay toggle (press 'U' to show)

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Time-based inference state
        self.inference_interval: float = 5.0
        self.last_inference_time: float = 0.0
        self.inference_cycle_count: int = 0

        # Overlay timing
        self.overlay_duration: float = 2.0
        self.overlay_start_time: float = 0.0
        self.overlay_active: bool = False

        # Danger assessment
        self.last_danger_score: float = 0.0
        self.current_danger_score: float = 0.0
        self.danger_trend: str = "stable"

        # Last response content (for manual replay)
        self.last_response_text: str = ""
        self.last_navigation_text: str = ""
        self.last_response_danger: float = 0.0

        # Smart mode response
        self.smart_response_text: str = ""
        self.smart_response_time: float = 0.0
        self.smart_response_duration: float = 2.0

        # Spatial understanding
        self.spatial_mode: bool = False
        self.spatial_frozen: bool = False
        self.spatial_frame: Optional[np.ndarray] = None
        self.spatial_detections = None
        self.spatial_description_text: str = ""
        self.spatial_description_display_time: float = 0.0
        self.spatial_description_timeout: float = 5.0

        # Deep spatial understanding (combines depth map + detections)
        self.deep_spatial_mode: bool = False
        self.deep_spatial_frozen: bool = False
        self.deep_spatial_frame: Optional[np.ndarray] = None
        self.deep_spatial_depth_map: Optional[np.ndarray] = None
        self.deep_spatial_combined_frame: Optional[np.ndarray] = None
        self.deep_spatial_detections = None
        self.deep_spatial_description_text: str = ""
        self.deep_spatial_description_display_time: float = 0.0

        # Help system
        self.show_help: bool = False

        # ArUco marker detection
        self.aruco_enabled: bool = False
        self.aruco_detections: List = []
        self.last_aruco_summary: str = ""
        self.aruco_summary_time: float = 0.0

        # Visualization options
        self.background_mode: bool = False
        self.visualize: bool = True
        self.csv_logging: bool = False

        # Model tracking
        self.enabled_models: List[str] = ["gpt-4o"]
        self.model_results: Dict[str, Any] = {}
        self.model_last_run: Dict[str, float] = {}

    def should_run_inference(self) -> bool:
        current_time = time.monotonic()
        return (current_time - self.last_inference_time) >= self.inference_interval

    def mark_inference_complete(self) -> None:
        self.last_inference_time = time.monotonic()
        self.inference_cycle_count += 1

    def get_time_until_next_inference(self) -> float:
        elapsed = time.monotonic() - self.last_inference_time
        remaining = self.inference_interval - elapsed
        return max(0.0, remaining)

    def should_show_overlay(self) -> bool:
        if not self.overlay_active:
            return False
        return (time.monotonic() - self.overlay_start_time) < self.overlay_duration

    def get_overlay_opacity(self) -> float:
        if not self.overlay_active:
            return 0.0
        elapsed = time.monotonic() - self.overlay_start_time
        remaining = self.overlay_duration - elapsed
        if remaining <= 0:
            return 0.0
        fade_start = 0.5
        if remaining <= fade_start:
            return remaining / fade_start
        return 1.0

    def start_overlay(self) -> None:
        self.overlay_start_time = time.monotonic()
        self.overlay_active = True

    def stop_overlay(self) -> None:
        self.overlay_active = False

    def update_inference_result(self, result: Dict[str, Any]) -> None:
        danger_score = result.get("danger_score", 0.0)
        reason = result.get("reason", "")
        navigation = result.get("navigation", "")

        self.last_danger_score = self.current_danger_score
        self.current_danger_score = danger_score
        if danger_score > self.last_danger_score + 0.1:
            self.danger_trend = "increasing"
        elif danger_score < self.last_danger_score - 0.1:
            self.danger_trend = "decreasing"
        else:
            self.danger_trend = "stable"

        # Store last response for manual replay
        self.last_response_text = reason or ""
        self.last_navigation_text = navigation or ""
        self.last_response_danger = danger_score
        self.start_overlay()

    def get_inference_stats(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.inference_cycle_count,
            "inference_interval": self.inference_interval,
            "time_until_next": self.get_time_until_next_inference(),
            "overlay_visible": self.should_show_overlay(),
            "overlay_opacity": self.get_overlay_opacity(),
            "danger_score": self.current_danger_score,
            "danger_trend": self.danger_trend,
        }


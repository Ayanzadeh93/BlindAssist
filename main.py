#!/usr/bin/env python3
"""AIDGPT main application with time-based inference (5s) and TTS output."""

import cv2
import numpy as np
import time
import threading
import queue
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from args import parse_arguments, get_argument_summary
from config import config
from detection_engine import DetectionEngine
from app_utils import get_config_value, create_placeholder_frame
try:
    from detection_engine_onnx import DetectionEngineONNX
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    DetectionEngineONNX = None
from gpt_interface import GPTInterface
from source_handler import VideoSource
from help import HelpSystem
from model_manager import get_model_manager, get_available_models
from metrics import PerformanceMetrics, CSVLogger
from application_state import ApplicationState
from audio_system import AudioPriority, PriorityAudioQueue, TTSManager
from overlays import add_overlay, add_controls_overlay, add_spatial_overlay, add_deep_spatial_overlay, add_aruco_overlay
from spatial_understanding import generate_spatial_description
from deep_spatial_understanding import (
    create_combined_frame, 
    generate_deep_spatial_description
)
from frame_processor import categorize_detections, DepthEstimator
from input_handler import handle_keyboard, create_mouse_callback
from inference_scheduler import (
    InferenceScheduler, GPTModelAdapter, DepthModelAdapter,
    InferenceResult, get_inference_scheduler
)
from overlay_manager import OverlayManager, get_overlay_manager
from tts_scheduler import get_tts_scheduler
from aruco_detector import ArucoDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)


class Application:
    def __init__(self, args: argparse.Namespace) -> None:
        self.setup_application(args)
        self.initialize_components()
        self.setup_time_based_scheduler()
        self.setup_queues()
        self.setup_windows()
        logging.info("Application initialized with time-based inference")

    def setup_application(self, args: argparse.Namespace) -> None:
        """Initialize application state and configuration."""
        self._setup_state_and_config(args)
        self._setup_model_manager(args)
        self._setup_video_source(args)
        self._setup_csv_logging(args)

    def _setup_state_and_config(self, args: argparse.Namespace) -> None:
        """Initialize application state and configuration values."""
        self.state = ApplicationState()
        self.frame_id = 0
        self.state.inference_interval = get_config_value(args, 'inference_interval', 'INFERENCE_INTERVAL', 5.0)
        self.state.overlay_duration = get_config_value(args, 'overlay_duration', 'OVERLAY_DURATION', 2.0)
        self._spatial_display_frame = None
        self._deep_spatial_display_frame = None
        self.state.background_mode = args.background_mode
        self.state.visualize = args.visualize and not args.no_visualize
        self.state.csv_logging = args.log_csv

    def _setup_model_manager(self, args: argparse.Namespace) -> None:
        """Initialize model manager and AI model."""
        self.model_manager = get_model_manager()
        self.ai_model = getattr(args, 'ai_model', 'gpt-4o')
        if not self.model_manager.set_current_model(self.ai_model):
            logging.warning(f"Model {self.ai_model} not available, using fallback")
            self.ai_model = self.model_manager.get_current_model()
        logging.info(f"Using AI model: {self.ai_model}")
        logging.info(f"Available models: {get_available_models()}")
        logging.info(f"Inference interval: {self.state.inference_interval}s")
        logging.info(f"Overlay duration: {self.state.overlay_duration}s")

    def _setup_video_source(self, args: argparse.Namespace) -> None:
        """Setup video source from arguments."""
        try:
            input_source = self._determine_input_source(args)
            output_path = Path(args.output) if args.output else self.generate_output_path()
            self.video_source = VideoSource(input_source, output_path)
            self.frame_width, self.frame_height = self.video_source.get_dimensions()
            self._log_video_source_info(input_source)
        except Exception:
            logging.exception("Error setting up video source")
            raise

    def _determine_input_source(self, args: argparse.Namespace):
        """Determine input source from arguments."""
        if hasattr(args, 'phone_camera') and args.phone_camera:
            input_source = self._build_phone_camera_url(args.phone_camera)
            logging.info(f"Using phone camera: {input_source}")
            return input_source
        elif args.video_file:
            logging.info(f"Processing video file: {args.video_file}")
            return args.video_file
        elif args.input and args.input.isdigit():
            input_source = int(args.input)
            logging.info(f"Using camera input (device {input_source})")
            return input_source
        else:
            input_source = args.input if args.input else 0
            logging.info(f"Using input source: {input_source}")
            return input_source

    def _log_video_source_info(self, input_source) -> None:
        """Log video source information."""
        if isinstance(input_source, str):
            logging.info(f"Video file: {input_source}")
            logging.info(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        else:
            logging.info(f"Camera device: {input_source}")

    def _setup_csv_logging(self, args: argparse.Namespace) -> None:
        """Setup CSV logging if enabled."""
        self.gpt_cooldown = args.gpt_cooldown
        self.keep_days = args.keep_days
        self.batch_size = args.batch_size
        self.processing_interval = args.processing_interval
        self.video_file = args.video_file
        self.csv_logger = CSVLogger(args.csv_output, args.log_interval) if self.state.csv_logging else None

    def _build_phone_camera_url(self, phone_input: str) -> str:
        phone_input = phone_input.strip()
        if phone_input.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
            return phone_input
        if ':' in phone_input and not phone_input.startswith('http'):
            ip, port = phone_input.split(':', 1)
            logging.info(f"Trying phone camera formats for {ip}:{port}")
            return f"http://{ip}:{port}/video"
        logging.info(f"Auto-detecting phone camera URL for {phone_input}")
        return f"http://{phone_input}:8080/video"

    def initialize_components(self) -> None:
        try:
            logging.info("Initializing detection engine...")
            use_gpu = get_config_value(None, None, 'USE_GPU', False)
            if use_gpu:
                try:
                    logging.info("Attempting native GPU...")
                    self.detection_engine = DetectionEngine(device='cuda')
                except Exception as e:
                    logging.warning(f"Failed GPU engine: {e}")
                    if ONNX_AVAILABLE:
                        try:
                            logging.info("Attempting ONNX DirectML GPU engine...")
                            self.detection_engine = DetectionEngineONNX()
                        except Exception as onnx_error:
                            logging.warning(f"Failed ONNX GPU engine: {onnx_error}")
                            config.USE_GPU = False
                            config.DEVICE = 'cpu'
                            self.detection_engine = DetectionEngine(device='cpu')
                    else:
                        config.USE_GPU = False
                        config.DEVICE = 'cpu'
                        self.detection_engine = DetectionEngine(device='cpu')
            else:
                config.DEVICE = 'cpu'
                self.detection_engine = DetectionEngine(device='cpu')

            logging.info("Initializing GPT interface...")
            self.gpt_interface = GPTInterface()
            self.help_system = HelpSystem()

            logging.info("Initializing TTS engine...")
            self.tts_manager = TTSManager(audio_enabled=True)
            self.state.audio_enabled = bool(self.tts_manager.audio_enabled)

            logging.info("Setting up Depth Anything V2 (lazy load)...")
            self.depth_estimator = DepthEstimator()
            if self.depth_estimator.is_available():
                logging.info("✓ Depth Anything V2 ready")
            else:
                logging.info("⚠ Depth Anything V2 will load on demand")

            logging.info("Initializing ArUco marker detector...")
            self.aruco_detector = ArucoDetector(
                marker_size_m=0.15,  # 15cm markers on A4 paper
                dictionary_type=None,  # None = auto-detect dictionary
                auto_detect_dictionary=True  # Try multiple dictionaries automatically
            )
            logging.info("✓ ArUco detector ready (auto-detect mode enabled)")
        except Exception:
            logging.exception("Error initializing components")
            raise

    def setup_time_based_scheduler(self) -> None:
        logging.info("Setting up time-based inference scheduler...")
        inference_interval = get_config_value(None, None, 'INFERENCE_INTERVAL', 5.0)
        overlay_duration = get_config_value(None, None, 'OVERLAY_DURATION', 2.0)
        self.inference_scheduler = get_inference_scheduler(
            inference_interval=inference_interval,
            audio_queue=None
        )
        self.overlay_manager = get_overlay_manager(overlay_duration)
        # Keep scheduler audio queueing aligned with current audio toggle
        self.inference_scheduler.set_audio_enabled_getter(lambda: self.state.audio_enabled)
        gpt_adapter = GPTModelAdapter(self.gpt_interface, self.ai_model)
        self.inference_scheduler.register_model(gpt_adapter, enabled=True)
        if self.state.show_depth_map:
            depth_adapter = DepthModelAdapter(self.depth_estimator)
            self.inference_scheduler.register_model(depth_adapter, enabled=True)
        self.inference_scheduler.add_result_callback(self._on_inference_complete)
        logging.info(f"Scheduler initialized: {inference_interval}s interval, {overlay_duration}s overlay")

    def _on_inference_complete(self, results: Dict[str, InferenceResult]) -> None:
        if not results:
            return
        primary_result = self.inference_scheduler.get_primary_response()
        if primary_result:
            self.overlay_manager.update_from_inference_result(primary_result)
            self.state.update_inference_result({
                "danger_score": primary_result.danger_score,
                "reason": primary_result.response_text,
                "navigation": primary_result.navigation
            })
            logging.info(f"Inference cycle {self.inference_scheduler.cycle_count}: danger={primary_result.danger_score:.2f}, model={primary_result.model_name}")

    def setup_queues(self) -> None:
        self.audio_queue = PriorityAudioQueue(maxsize=10)
        self.inference_scheduler.audio_queue = self.audio_queue
        self.tts_scheduler = get_tts_scheduler(
            self.tts_manager,
            self.audio_queue,
            self.state.inference_interval
        )
        self.gpt_queue = queue.Queue(maxsize=1)
        self.gpt_result_queue = queue.Queue(maxsize=1)

    def setup_windows(self) -> None:
        """Setup visualization windows."""
        try:
            if self.state.background_mode or not self.state.visualize:
                self.windows_initialized = False
                logging.info("Background mode - no visualization windows")
                return
            
            cv2.namedWindow('Main Feed', cv2.WINDOW_NORMAL)
            width, height = self._calculate_window_size()
            self._configure_window(width, height)
            self.windows_initialized = True
            self.last_depth_state = False
            logging.info("Window setup completed successfully")
        except Exception:
            logging.exception("Error setting up windows")
            self.windows_initialized = False

    def _calculate_window_size(self) -> Tuple[int, int]:
        """Calculate optimal window size based on source dimensions."""
        source_width = getattr(self, 'frame_width', None)
        source_height = getattr(self, 'frame_height', None)
        
        if source_width and source_height:
            aspect_ratio = source_width / source_height
            screen_width, screen_height = self._get_screen_size()
            max_width = int(screen_width * 0.9)
            max_height = int(screen_height * 0.9)
            
            if source_width > max_width or source_height > max_height:
                scale = min(max_width / source_width, max_height / source_height)
                width = int(source_width * scale)
                height = int(source_height * scale)
            else:
                width, height = source_width, source_height
            
            # Ensure minimum size
            min_width, min_height = 320, 240
            if width < min_width:
                width = min_width
                height = int(width / aspect_ratio)
            if height < min_height:
                height = min_height
                width = int(height * aspect_ratio)
            
            return width, height
        else:
            return get_config_value(None, None, 'FRAME_WIDTH', 1280), get_config_value(None, None, 'FRAME_HEIGHT', 720)

    def _configure_window(self, width: int, height: int) -> None:
        """Configure window position and properties."""
        cv2.resizeWindow('Main Feed', width, height)
        try:
            screen_width, screen_height = self._get_screen_size()
            x_pos = (screen_width - width) // 2
            y_pos = (screen_height - height) // 2
            cv2.moveWindow('Main Feed', max(0, x_pos), max(0, y_pos))
        except Exception:
            pass
        cv2.setWindowProperty('Main Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Main Feed', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
        mouse_callback = create_mouse_callback(self.state, self.help_system)
        cv2.setMouseCallback('Main Feed', mouse_callback)

    def _get_screen_size(self) -> Tuple[int, int]:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return width, height
            except Exception:
                logging.warning("Could not detect screen size, using defaults")
                return 1920, 1080

    def _create_error_frame(self, message: str) -> np.ndarray:
        """Create an error frame with message."""
        return create_placeholder_frame(480, 640, message, (0, 255, 255))

    def _prepare_display_frame(self, annotated_frame: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
        """Prepare frame for display, handling side-by-side modes."""
        # Validate frames
        if annotated_frame is None or annotated_frame.size == 0:
            annotated_frame = original_frame if original_frame is not None and original_frame.size > 0 else None
        if annotated_frame is None:
            return self._create_error_frame("No frame available")
        if original_frame is None or original_frame.size == 0:
            return annotated_frame

        try:
            # Handle window resizing on mode changes
            self._update_window_for_mode_changes()
            
            h, w = annotated_frame.shape[:2]
            
            # Deep spatial mode (frozen combined frame)
            if self.state.deep_spatial_mode and self.state.deep_spatial_frozen:
                return self._get_deep_spatial_frame(annotated_frame)
            
            # Spatial mode (frozen frame + description)
            if self.state.spatial_mode:
                spatial_frame = getattr(self, '_spatial_display_frame', None)
                if spatial_frame is not None and spatial_frame.size > 0:
                    return self._create_spatial_side_by_side(annotated_frame, spatial_frame, h, w)
            
            # Depth mode (original + depth map)
            if self.state.show_depth_map:
                return self._create_depth_side_by_side(annotated_frame, original_frame, h, w)
                
            return annotated_frame
        except Exception:
            logging.exception("Error preparing display frame")
            return annotated_frame if annotated_frame is not None else self._create_error_frame("Display error")

    def _update_window_for_mode_changes(self) -> None:
        """Check mode changes and resize window if needed."""
        last_depth = getattr(self, 'last_depth_state', False)
        last_spatial = getattr(self, 'last_spatial_state', False)
        last_deep_spatial = getattr(self, 'last_deep_spatial_state', False)
        
        if (last_depth != self.state.show_depth_map or 
            last_spatial != self.state.spatial_mode or 
            last_deep_spatial != self.state.deep_spatial_mode):
            self._resize_window_for_side_by_side()
            self.last_depth_state = self.state.show_depth_map
            self.last_spatial_state = self.state.spatial_mode
            self.last_deep_spatial_state = self.state.deep_spatial_mode

    def _get_deep_spatial_frame(self, fallback: np.ndarray) -> np.ndarray:
        """Get deep spatial display frame with fallback."""
        deep_frame = getattr(self, '_deep_spatial_display_frame', None)
        if deep_frame is not None and deep_frame.size > 0:
            return deep_frame
        combined = self.state.deep_spatial_combined_frame
        if combined is not None and combined.size > 0:
            return combined
        return fallback

    def _create_spatial_side_by_side(self, annotated_frame: np.ndarray, spatial_panel: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create side-by-side view for spatial mode."""
        sp_h, sp_w = spatial_panel.shape[:2]
        if sp_h != h:
            spatial_panel = cv2.resize(spatial_panel, (int(sp_w * h / sp_h), h))
        combined = np.hstack([annotated_frame, spatial_panel])
        cv2.putText(combined, "Frozen Frame", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined, "Spatial Description", (w + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        return combined

    def _create_depth_side_by_side(self, annotated_frame: np.ndarray, original_frame: np.ndarray, h: int, w: int) -> np.ndarray:
        """Create side-by-side view for depth mode."""
        depth_map = self.depth_estimator.estimate_depth(original_frame)
        if depth_map is None or depth_map.size == 0:
            return annotated_frame
        if len(depth_map.shape) == 2:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        depth_h, depth_w = depth_map.shape[:2]
        if depth_h != h:
            depth_map = cv2.resize(depth_map, (int(depth_w * h / depth_h), h))
        combined = np.hstack([annotated_frame, depth_map])
        cv2.putText(combined, "Original", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Depth Map", (w + 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return combined

    def _resize_window_for_side_by_side(self) -> None:
        """Resize window for side-by-side display (depth map or spatial mode)."""
        try:
            if not self.windows_initialized:
                return
            source_width = getattr(self, 'frame_width', None)
            source_height = getattr(self, 'frame_height', None)
            if not source_width or not source_height:
                return
            if self.state.show_depth_map or self.state.spatial_mode or self.state.deep_spatial_mode:
                new_width = source_width * 2
                new_height = source_height
                screen_width, screen_height = self._get_screen_size()
                max_width = int(screen_width * 0.9)
                max_height = int(screen_height * 0.9)
                if new_width > max_width or new_height > max_height:
                    scale_w = max_width / new_width
                    scale_h = max_height / new_height
                    scale = min(scale_w, scale_h)
                    new_width = int(new_width * scale)
                    new_height = int(new_height * scale)
                cv2.resizeWindow('Main Feed', new_width, new_height)
                logging.info(f"Window resized for depth map: {new_width}x{new_height}")
            else:
                screen_width, screen_height = self._get_screen_size()
                max_width = int(screen_width * 0.8)
                max_height = int(screen_height * 0.8)
                aspect_ratio = source_width / source_height
                if source_width > max_width or source_height > max_height:
                    scale_w = max_width / source_width
                    scale_h = max_height / source_height
                    scale = min(scale_w, scale_h)
                    width = int(source_width * scale)
                    height = int(source_height * scale)
                else:
                    width = source_width
                    height = source_height
                cv2.resizeWindow('Main Feed', width, height)
                logging.info(f"Window restored: {width}x{height}")
        except Exception:
            logging.exception("Error resizing window for depth map")

    def _create_help_window(self) -> np.ndarray:
        """Create a help window frame (displayed in separate window)."""
        # Create a dark background
        h, w = 500, 600
        help_frame = np.zeros((h, w, 3), dtype=np.uint8)
        help_frame[:] = (30, 30, 30)  # Dark gray background
        
        # Help content
        help_sections = [
            ("AIDGPT HELP", (0, 255, 255), 0.9, 2),
            ("", None, 0, 0),
            ("KEYBOARD CONTROLS", (60, 180, 255), 0.6, 2),
            ("  Q - Quit", (255, 255, 255), 0.5, 1),
            ("  P - Pause/Resume", (255, 255, 255), 0.5, 1),
            ("  D - Toggle debug", (255, 255, 255), 0.5, 1),
            ("  M - Toggle depth map", (255, 255, 255), 0.5, 1),
            ("  B - Toggle detections", (255, 255, 255), 0.5, 1),
            ("  G - Toggle GPT overlay", (255, 255, 255), 0.5, 1),
            ("  A - Toggle audio", (255, 255, 255), 0.5, 1),
            ("  T - Test audio", (255, 255, 255), 0.5, 1),
            ("  V - Toggle alerts", (255, 255, 255), 0.5, 1),
            ("  S - Smart mode", (255, 255, 255), 0.5, 1),
            ("  K - Spatial understanding", (255, 255, 255), 0.5, 1),
            ("  J - Deep spatial (depth+detect)", (255, 255, 255), 0.5, 1),
            ("  R - Toggle ArUco detection", (255, 255, 255), 0.5, 1),
            ("  H - Toggle help", (255, 255, 255), 0.5, 1),
            ("  U - Toggle UI panel", (255, 255, 255), 0.5, 1),
            ("", None, 0, 0),
            ("MOUSE CONTROLS", (60, 180, 255), 0.6, 2),
            ("  Left Click - Debug toggle", (255, 255, 255), 0.5, 1),
            ("  Right Click - Pause", (255, 255, 255), 0.5, 1),
            ("  Scroll - Navigate", (255, 255, 255), 0.5, 1),
            ("", None, 0, 0),
            ("Press H to close", (0, 255, 0), 0.6, 2),
        ]
        
        y = 35
        for text, color, scale, thickness in help_sections:
            if color is None:
                y += 10
                continue
            cv2.putText(help_frame, text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += 22
        
        # Draw border
        cv2.rectangle(help_frame, (5, 5), (w-5, h-5), (60, 180, 255), 2)
        
        return help_frame

    def generate_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(config.OUTPUT_DIR) / f"processed_video_{timestamp}.mp4"

    def _process_spatial_mode(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Handle spatial mode processing."""
        if not self.state.spatial_frozen:
            # First time - capture and FREEZE this frame
            self.state.spatial_frame = frame.copy()
            detections, detected_frame = self.detection_engine.detect_and_track(frame)
            self.state.spatial_detections = detections
            annotated_frame = detected_frame if detections else frame.copy()
            self.state.spatial_frame = annotated_frame.copy()
            generate_spatial_description(self.state, self.gpt_interface, self.audio_queue)
            self._spatial_display_frame = annotated_frame.copy()
            self._spatial_display_frame = add_spatial_overlay(self._spatial_display_frame, self.state)
            return annotated_frame, len(detections) if detections else 0
        
        # Return frozen frame
        if self.state.spatial_frame is not None:
            annotated_frame = self.state.spatial_frame.copy()
            self._spatial_display_frame = annotated_frame.copy()
            self._spatial_display_frame = add_spatial_overlay(self._spatial_display_frame, self.state)
            return annotated_frame, len(self.state.spatial_detections) if self.state.spatial_detections else 0
        return frame, 0

    def _process_deep_spatial_mode(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Handle deep spatial mode processing."""
        if self.state.deep_spatial_frozen:
            if self.state.deep_spatial_combined_frame is not None:
                annotated_frame = self.state.deep_spatial_combined_frame.copy()
                annotated_frame = add_deep_spatial_overlay(annotated_frame, self.state)
                self._deep_spatial_display_frame = annotated_frame.copy()
                return annotated_frame, len(self.state.deep_spatial_detections) if self.state.deep_spatial_detections else 0
            logging.warning("Deep spatial combined frame missing, disabling mode")
            self.state.deep_spatial_mode = False
            self.state.deep_spatial_frozen = False
            self._deep_spatial_display_frame = None
            return frame, 0
        
        # First time - freeze frame, get depth, create combined
        if not self.state.show_depth_map:
            logging.warning("Deep spatial requires depth map. Disabling.")
            self.state.deep_spatial_mode = False
            return frame, 0
        
        self.state.deep_spatial_frame = frame.copy()
        detections, detected_frame = self.detection_engine.detect_and_track(frame)
        self.state.deep_spatial_detections = detections
        annotated_frame = detected_frame if detections else frame.copy()
        
        depth_map = self.depth_estimator.estimate_depth(frame)
        if depth_map is None or depth_map.size == 0:
            logging.warning("Depth map not available, disabling deep spatial")
            self.state.deep_spatial_mode = False
            return annotated_frame, len(detections) if detections else 0
        
        self.state.deep_spatial_depth_map = depth_map
        combined = create_combined_frame(annotated_frame, depth_map)
        if combined is None or combined.size == 0:
            logging.error("Failed to create combined frame")
            self.state.deep_spatial_mode = False
            return annotated_frame, len(detections) if detections else 0
        
        self.state.deep_spatial_combined_frame = combined
        
        # Generate description in background
        def generate_async():
            try:
                generate_deep_spatial_description(
                    self.state, self.gpt_interface, self.audio_queue, self.depth_estimator
                )
            except Exception:
                pass
        
        threading.Thread(target=generate_async, daemon=True).start()
        self.state.deep_spatial_frozen = True
        
        combined_with_overlay = combined.copy()
        combined_with_overlay = add_deep_spatial_overlay(combined_with_overlay, self.state)
        self._deep_spatial_display_frame = combined_with_overlay.copy()
        return combined_with_overlay, len(detections) if detections else 0

    def _process_detections(self, frame: np.ndarray) -> Tuple[np.ndarray, List, int]:
        """Process object detections on frame (optimized)."""
        if not self.state.show_detections:
            return frame, [], 0
        
        try:
            raw_depth = None
            if self.state.show_depth_map and self.depth_estimator.is_available():
                try:
                    raw_depth = self.depth_estimator.get_raw_depth(frame)
                except Exception:
                    pass
            
            detections, detected_frame = self.detection_engine.detect_and_track(frame, raw_depth)
            if detections:
                return detected_frame, detections, len(detections)
            return frame, [], 0
        except Exception:
            return frame, [], 0

    def _process_aruco(self, annotated_frame: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Process ArUco marker detection."""
        if not self.state.aruco_enabled or annotated_frame is None:
            return annotated_frame
        
        try:
            is_gpt_processing = self.overlay_manager.is_processing()
            should_run_inference = hasattr(self, 'inference_scheduler') and self.inference_scheduler.should_run_inference()
            
            if not is_gpt_processing and not should_run_inference:
                if not hasattr(self, '_aruco_frame_skip'):
                    self._aruco_frame_skip = 0
                self._aruco_frame_skip += 1
                
                if self._aruco_frame_skip % 2 == 0:
                    try:
                        aruco_detections, frame_with_aruco = self.aruco_detector.detect(annotated_frame)
                        self.state.aruco_detections = aruco_detections
                        
                        if frame_with_aruco is not None and aruco_detections:
                            annotated_frame = frame_with_aruco
                            current_time = time.time()
                            if current_time - self.state.aruco_summary_time > 2.0:
                                summary = self.aruco_detector.get_detection_summary(aruco_detections)
                                self.state.last_aruco_summary = summary
                                self.state.aruco_summary_time = current_time
                                if self.state.audio_enabled:
                                    self.audio_queue.put_nowait(summary, AudioPriority.NORMAL)
                    except Exception:
                        pass
            
            annotated_frame = add_aruco_overlay(annotated_frame, self.state)
        except Exception:
            annotated_frame = frame if annotated_frame is None else annotated_frame
        
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Main frame processing method."""
        try:
            start_time = time.time()

            # Handle spatial mode
            if self.state.spatial_mode:
                return self._process_spatial_mode(frame)

            # Handle deep spatial mode
            if self.state.deep_spatial_mode:
                result = self._process_deep_spatial_mode(frame)
                # Clear display frame when mode is off
                if not self.state.deep_spatial_mode:
                    self._deep_spatial_display_frame = None
                return result
            else:
                # Clear deep spatial display frame when mode is off
                self._deep_spatial_display_frame = None

            # Process detections
            annotated_frame, detections, num_detections = self._process_detections(frame)

            # Apply debug overlay
            if self.state.show_debug and not isinstance(self.video_source.source, str):
                try:
                    annotated_frame = add_overlay(annotated_frame, self.state, self.frame_id)
                except Exception:
                    pass  # Continue without debug overlay

            # Process GPT inference (async - don't block frame processing!)
            # Only check and start inference once per interval to avoid overhead
            if self.state.show_gpt_overlay and self.inference_scheduler.should_run_inference():
                # Mark as processing immediately (non-blocking)
                self.overlay_manager.set_processing_state(True)
                
                # Run everything in background thread to avoid ANY blocking
                def run_inference_async():
                    try:
                        # Categorize detections in background (not in main thread)
                        frame_info = categorize_detections(
                            detections, self.frame_id, self.frame_width, self.frame_height
                        )
                        # Update frame in background (frame copy happens here, not blocking main thread)
                        self.inference_scheduler.update_frame(frame, detections, frame_info)
                        # Run inference cycle
                        self.inference_scheduler.run_inference_cycle()
                        self.state.metrics.gpt_calls += 1
                    except Exception:
                        pass
                    finally:
                        self.overlay_manager.set_processing_state(False)
                
                threading.Thread(target=run_inference_async, daemon=True).start()

            # Apply GPT overlays
            if self.state.show_gpt_overlay and annotated_frame is not None:
                annotated_frame = self.overlay_manager.render_overlays(annotated_frame)

            # Process ArUco markers
            annotated_frame = self._process_aruco(annotated_frame, frame)

            # Apply help overlay
            if self.state.show_help:
                annotated_frame = self.help_system.add_help_overlay(annotated_frame)

            # Apply controls overlay
            if not isinstance(self.video_source.source, str):
                if annotated_frame is None:
                    annotated_frame = frame
                if annotated_frame is not None:
                    annotated_frame = add_controls_overlay(annotated_frame, self.state)

            # Final validation - ensure we have a frame
            if annotated_frame is None:
                annotated_frame = frame

            processing_time = time.time() - start_time
            self.state.metrics.update(processing_time, num_detections)
            return annotated_frame, num_detections
        except Exception:
            return frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), 0

    def run(self) -> None:
        logging.info("Starting application with time-based inference...")
        logging.info(f"Inference interval: {self.state.inference_interval}s")
        logging.info(f"Overlay duration: {self.state.overlay_duration}s")
        try:
            if isinstance(self.video_source.source, str):
                if not os.path.exists(self.video_source.source):
                    raise FileNotFoundError(f"Video file not found: {self.video_source.source}")
                logging.info(f"Processing video file: {self.video_source.source}")
                logging.info(f"Video properties: Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.video_source.fps}")
            else:
                logging.info(f"Using camera input (device {self.video_source.source})")
        except Exception:
            logging.exception("Error verifying input source")
            return

        try:
            if self.tts_scheduler:
                self.tts_scheduler.start()
                logging.info("TTS scheduler worker started")
            else:
                logging.warning("TTS scheduler not available, starting fallback audio worker")
                audio_thread = threading.Thread(
                    target=self._audio_worker,
                    daemon=True
                )
                audio_thread.start()
                logging.info("Fallback audio worker thread started")
        except Exception:
            logging.exception("Error starting worker threads")
            return

        try:
            self.video_source.setup_writer()
            if self.csv_logger:
                self.csv_logger.set_video_start_time(time.time())
            while self.state.running:
                if self.state.paused:
                    if not self.state.background_mode:
                        if cv2.waitKey(1) & 0xFF == ord('p'):
                            self.state.paused = False
                    continue

                ret, frame = self.video_source.read()
                if not ret:
                    if isinstance(self.video_source.source, str):
                        logging.info("End of video file reached")
                    else:
                        logging.error("Failed to read frame from camera")
                    break

                self.frame_id += 1
                try:
                    start_time = time.time()
                    annotated_frame, num_detections = self.process_frame(frame)
                    processing_time = time.time() - start_time
                    self.state.metrics.update(processing_time, num_detections)
                    if not self.state.background_mode and self.state.visualize:
                        display_frame = self._prepare_display_frame(annotated_frame, frame)
                        cv2.imshow('Main Feed', display_frame)
                        
                        # Show help in separate window (like depth map)
                        if self.state.show_help:
                            help_frame = self._create_help_window()
                            cv2.imshow('Help - Controls', help_frame)
                        else:
                            try:
                                cv2.destroyWindow('Help - Controls')
                            except:
                                pass
                    self.video_source.write(annotated_frame)
                    if isinstance(self.video_source.source, str) and self.frame_id % 30 == 0:
                        logging.info(f"Processed frame {self.frame_id}")
                except Exception:
                    logging.exception(f"Error processing frame {self.frame_id}")
                    continue

                if not self.state.background_mode:
                    is_camera = isinstance(self.video_source.source, int)
                    if not handle_keyboard(self.state, self.help_system, self.audio_queue, self.tts_manager, is_camera):
                        break
                if isinstance(self.video_source.source, int):
                    if self.video_source.cap:
                        for _ in range(2):
                            self.video_source.cap.grab()
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        except Exception:
            logging.exception("Unexpected error in main loop")
        finally:
            try:
                self.cleanup()
                logging.info("Application shutdown complete")
            except Exception:
                logging.exception("Error during cleanup")

    def _audio_worker(self) -> None:
        logging.info("Audio worker thread started")
        while self.state.running:
            try:
                audio_msg = self.audio_queue.get(timeout=1.0)
                if audio_msg:
                    self.tts_manager.speak(audio_msg.text, audio_msg.bypass_cooldown)
            except queue.Empty:
                continue
            except Exception:
                logging.exception("Error in audio worker")

    def cleanup(self) -> None:
        logging.info("Starting cleanup...")
        self.state.running = False
        try:
            if hasattr(self, 'tts_scheduler') and self.tts_scheduler:
                self.tts_scheduler.stop()
            if self.tts_manager:
                self.tts_manager.stop()
            self.video_source.release()
            if not self.state.background_mode:
                cv2.destroyAllWindows()
                for _ in range(5):
                    cv2.waitKey(1)
            if self.csv_logger:
                self.csv_logger.close()
        except Exception:
            logging.exception("Error during cleanup")
        self.print_final_statistics()

    def print_final_statistics(self) -> None:
        try:
            metrics = self.state.metrics
            inference_stats = self.state.get_inference_stats()
            logging.info(f"""
Final Statistics:
- Total Frames: {metrics.frame_count}
- Average FPS: {metrics.fps:.2f}
- Total Detections: {metrics.detection_count}
- Total Processing Time: {metrics.processing_time:.2f}s
- Average Detections per Frame: {metrics.detection_count/max(metrics.frame_count, 1):.2f}
- Total Inference Cycles: {inference_stats['cycle_count']}
- Inference Interval: {inference_stats['inference_interval']}s
- Total GPT Calls: {metrics.gpt_calls}
""")
        except Exception:
            logging.exception("Error printing statistics")


if __name__ == "__main__":
    args = parse_arguments()
    logging.info(get_argument_summary(args))
    try:
        app = Application(args)
        app.run()
    except KeyboardInterrupt:
        logging.info("Application terminated by user")
    except Exception:
        logging.exception("Unexpected error")
    finally:
        if 'app' in locals():
            app.cleanup()
            logging.info("Application cleanup complete")


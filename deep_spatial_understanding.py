#!/usr/bin/env python3
"""Deep spatial understanding module for AIDGPT application.

Combines camera feed with depth map to provide distance-aware spatial descriptions
using vision-capable LLM models with real depth analysis.
"""
import logging
import time
import numpy as np
import cv2
import os
import tempfile
from typing import Optional, List, Dict, Any
from gpt_interface import GPTInterface
from audio_system import AudioPriority, PriorityAudioQueue
from application_state import ApplicationState
from depth_analyzer import DepthAnalyzer, create_depth_analyzer


def toggle_deep_spatial_understanding(state: ApplicationState,
                                     audio_queue: PriorityAudioQueue,
                                     is_camera_input: bool = True) -> None:
    """Toggle deep spatial understanding mode.
    
    Prerequisites:
    - Depth map must be active (state.show_depth_map == True)
    - Must be using camera input (not video file)
    
    Args:
        state: ApplicationState instance
        audio_queue: PriorityAudioQueue instance for audio feedback
        is_camera_input: Whether input is from camera (True) or video file (False)
    """
    try:
        if not state.show_depth_map:
            audio_queue.put_nowait(
                "Deep spatial requires depth map. Please enable depth map first with M key.",
                AudioPriority.HIGH,
                bypass_cooldown=True
            )
            logging.warning("Deep spatial mode requires depth map to be active")
            return
        
        if not is_camera_input:
            audio_queue.put_nowait(
                "Deep spatial is only available with camera input.",
                AudioPriority.HIGH,
                bypass_cooldown=True
            )
            logging.warning("Deep spatial mode requires camera input")
            return
        
        if not state.deep_spatial_mode:
            # Activate deep spatial mode
            state.deep_spatial_mode = True
            state.deep_spatial_frozen = False  # Will freeze on next frame
            logging.info("Deep spatial understanding mode activated")
            audio_queue.put_nowait(
                "Deep spatial mode activated. Analyzing scene with depth.",
                AudioPriority.HIGH,
                bypass_cooldown=True
            )
        else:
            # Deactivate deep spatial mode - clear ALL state
            state.deep_spatial_mode = False
            state.deep_spatial_frozen = False
            state.deep_spatial_frame = None
            state.deep_spatial_depth_map = None
            state.deep_spatial_combined_frame = None
            state.deep_spatial_detections = None
            state.deep_spatial_description_text = ""
            logging.info("Deep spatial understanding mode deactivated")
            audio_queue.put_nowait(
                "Deep spatial mode deactivated",
                AudioPriority.NORMAL,
                bypass_cooldown=True
            )
    except Exception:
        logging.exception("Error toggling deep spatial understanding")


def create_combined_frame(annotated_frame: np.ndarray, depth_map: np.ndarray) -> Optional[np.ndarray]:
    """Create side-by-side composite of camera feed and depth map.
    
    Args:
        annotated_frame: Camera frame with bounding boxes drawn
        depth_map: Depth map visualization (colormap)
    
    Returns:
        Combined image (width = 2x original, height = original) or None on error
    """
    try:
        if annotated_frame is None or annotated_frame.size == 0:
            logging.error("Invalid annotated_frame for combined frame")
            return None
        if depth_map is None or depth_map.size == 0:
            logging.error("Invalid depth_map for combined frame")
            return None
            
        h1, w1 = annotated_frame.shape[:2]
        h2, w2 = depth_map.shape[:2]
        
        # Ensure depth map is 3-channel (BGR)
        if len(depth_map.shape) == 2:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        elif len(depth_map.shape) == 3 and depth_map.shape[2] == 1:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        
        # Resize depth map to match camera frame height
        if h2 != h1:
            new_w2 = int(w2 * h1 / h2)
            depth_map = cv2.resize(depth_map, (new_w2, h1), interpolation=cv2.INTER_LINEAR)
        
        # Create side-by-side composite
        combined = np.hstack([annotated_frame, depth_map])
        
        # Add labels at top
        label_y = 30
        cv2.putText(combined, "Camera Feed", (10, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (w1 + 10, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return combined
    except Exception:
        logging.exception("Error creating combined frame")
        return None


def extract_depth_for_detections(detections: List[Any], depth_map: np.ndarray, 
                                frame_width: int, depth_analyzer: Optional[DepthAnalyzer] = None) -> List[Dict[str, Any]]:
    """Extract depth information for each detection using DepthAnalyzer.
    
    Args:
        detections: List of detection objects with bbox coordinates
        depth_map: Depth map (resized to match camera frame height)
        frame_width: Width of original camera frame
        depth_analyzer: DepthAnalyzer instance (created if None)
    
    Returns:
        List of detection info dictionaries with depth data
    """
    detection_info = []
    
    try:
        if detections is None or len(detections) == 0:
            return detection_info
        
        # Create depth analyzer if not provided
        if depth_analyzer is None:
            depth_analyzer = create_depth_analyzer(max_distance=10.0)
        
        for detection in detections:
            try:
                x1, y1, x2, y2 = detection.bbox
                
                # Analyze depth for this bounding box
                depth_stats = depth_analyzer.analyze_region(depth_map, (x1, y1, x2, y2), frame_width)
                
                if depth_stats is None:
                    continue
                
                # Determine position category (left, center, right)
                center_x = (x1 + x2) // 2
                frame_third = frame_width / 3
                if center_x < frame_third:
                    position = "left"
                    clock = "9 o'clock"
                elif center_x < 2 * frame_third:
                    position = "center"
                    clock = "12 o'clock"
                else:
                    position = "right"
                    clock = "3 o'clock"
                
                distance_avg = depth_stats.mean_distance_m
                distance_category = depth_analyzer.get_distance_category(distance_avg)
                
                detection_info.append({
                    'class_name': detection.class_name,
                    'confidence': float(detection.confidence),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (int(center_x), int((y1 + y2) // 2)),
                    'position': position,
                    'clock_direction': clock,
                    'distance_avg': distance_avg,
                    'distance_min': depth_stats.min_distance_m,
                    'distance_max': depth_stats.max_distance_m,
                    'distance_center': depth_stats.center_distance_m,
                    'distance_category': distance_category,
                    'depth_confidence': depth_stats.confidence
                })
            except Exception:
                logging.exception(f"Error analyzing detection: {detection.class_name}")
                continue
    except Exception:
        logging.exception("Error extracting depth for detections")
    
    return detection_info


def depth_to_distance_category(distance_meters: float) -> str:
    """Convert distance in meters to descriptive category."""
    if distance_meters < 1.0:
        return "very close"
    elif distance_meters < 2.0:
        return "close"
    elif distance_meters < 4.0:
        return "near"
    elif distance_meters < 6.0:
        return "medium"
    elif distance_meters < 10.0:
        return "far"
    else:
        return "very far"


def generate_deep_spatial_description(state: ApplicationState,
                                     gpt_interface: GPTInterface,
                                     audio_queue: PriorityAudioQueue,
                                     depth_estimator=None) -> None:
    """Generate deep spatial description using composite image, vision API, and depth analysis.
    
    Args:
        state: ApplicationState instance
        gpt_interface: GPTInterface instance for querying vision API
        audio_queue: PriorityAudioQueue instance for audio feedback
        depth_estimator: DepthEstimator instance (optional, not used currently)
    """
    try:
        # Validate prerequisites
        if state.deep_spatial_frame is None:
            logging.warning("Cannot generate deep spatial: no frozen frame")
            state.deep_spatial_description_text = "No frame captured"
            return
        
        if state.deep_spatial_depth_map is None:
            logging.warning("Cannot generate deep spatial: no depth map")
            state.deep_spatial_description_text = "No depth map available"
            return
        
        if state.deep_spatial_combined_frame is None:
            logging.warning("Cannot generate deep spatial: no combined frame")
            state.deep_spatial_description_text = "No combined frame available"
            return
        
        # Create depth analyzer
        depth_analyzer = create_depth_analyzer(max_distance=10.0, debug=False)
        
        # Analyze overall scene depth
        scene_stats = depth_analyzer.analyze_scene_depth(state.deep_spatial_depth_map)
        logging.info(f"Scene depth analysis: closest={scene_stats.get('min_distance_m', 0)}m, "
                    f"avg={scene_stats.get('mean_distance_m', 0)}m, "
                    f"safety={scene_stats.get('safety_level', 'unknown')}")
        
        # Get detections and extract depth info with analyzer
        detections = state.deep_spatial_detections or []
        frame_width = state.deep_spatial_frame.shape[1]
        detection_info = extract_depth_for_detections(
            detections,
            state.deep_spatial_depth_map,
            frame_width,
            depth_analyzer
        )
        
        # Sort by distance (closest first)
        detection_info.sort(key=lambda x: x['distance_avg'])
        
        # Format depth information for LLM
        depth_info_text = depth_analyzer.format_depth_info_for_llm(detection_info, scene_stats)
        logging.info(f"Depth info prepared for LLM ({len(detection_info)} objects)")
        
        # Simplified prompt with depth analysis
        deep_spatial_prompt = f"""You are helping a blind user navigate. Analyze this composite image showing:
- LEFT: Camera view with detected objects (red boxes)
- RIGHT: Depth map (brighter colors = closer, darker = farther)

{depth_info_text}

TASK: Provide a clear, spoken navigation description (2-4 sentences) that:
1. Identifies main objects and positions (use clock directions: 12=ahead, 3=right, 9=left)
2. States actual distances from depth analysis (in meters)
3. Prioritizes closest/most important objects for navigation
4. Gives specific navigation advice if hazards are close (<2m)

Example format:
"There's a chair at 3 o'clock, 1.5 meters away - very close, watch out on your right. A table is straight ahead at 12 o'clock, 3.2 meters away. The path on your left at 9 o'clock appears clear beyond 5 meters."

Now describe what you see with accurate distances:"""
        
        # Save combined frame for vision API
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f'deep_spatial_{int(time.time())}.jpg')
        
        success = cv2.imwrite(temp_image_path, state.deep_spatial_combined_frame, 
                             [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        if not success or not os.path.exists(temp_image_path):
            logging.error("Failed to save composite image for vision API")
            state.deep_spatial_description_text = "Error: Could not save image"
            audio_queue.put_nowait("Error saving image for analysis", AudioPriority.HIGH, True)
            return
        
        file_size = os.path.getsize(temp_image_path)
        logging.info(f"Saved composite image: {temp_image_path} ({file_size} bytes)")
        
        # Call vision API with composite image and depth analysis
        logging.info("Calling GPT-4o Vision API with depth analysis...")
        gpt_response = gpt_interface.analyze_image(
            temp_image_path,
            deep_spatial_prompt,
            max_tokens=350,
            model="gpt-4o"
        )
        
        # Clean up temp file
        try:
            os.remove(temp_image_path)
        except Exception:
            pass
        
        if gpt_response and len(gpt_response.strip()) > 0:
            state.deep_spatial_description_text = gpt_response
            state.deep_spatial_description_display_time = time.monotonic()
            
            # Queue description with high priority
            audio_queue.put_nowait(
                gpt_response,
                AudioPriority.HIGH,
                bypass_cooldown=True
            )
            logging.info(f"Deep spatial description generated: '{gpt_response[:100]}...'")
        else:
            state.deep_spatial_description_text = "Unable to generate description"
            audio_queue.put_nowait(
                "Unable to analyze scene",
                AudioPriority.NORMAL,
                bypass_cooldown=True
            )
            logging.warning("Vision API returned empty response")
            
    except Exception:
        logging.exception("Error generating deep spatial description")
        state.deep_spatial_description_text = "Error during analysis"
        audio_queue.put_nowait("Error analyzing scene", AudioPriority.HIGH, True)

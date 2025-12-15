#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from config import config
from app_utils import get_config_value, validate_frame
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class Detection:
    """Data class for storing detection information."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    tracking_id: Optional[int] = None
    movement: Optional[Dict] = None

class DetectionEngine:
    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize the detection engine with the YOLO model.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        try:
            weight_path = get_config_value(None, None, 'weight_path', None)
            if not weight_path:
                raise ValueError("No weight_path defined in config.")
            
            # Get device from config or use provided device
            if device is None:
                device = get_config_value(None, None, 'DEVICE', 'cpu')
            
            # Allow using model id (e.g., 'yolo12n.pt') if local file is missing
            model_source = str(weight_path) if os.path.exists(weight_path) else os.path.basename(str(weight_path))

            logging.info(f"Loading YOLO model from: {model_source}")
            logging.info(f"Using device: {device}")
            
            # Load model on specified device
            self.model = YOLO(model_source)
            if device == 'cuda':
                # YOLO automatically uses GPU if available, but we can explicitly set it
                self.model.to(device)
                logging.info(f"Model loaded on GPU: {device}")
            else:
                logging.info(f"Model loaded on CPU")
            
            self.device = device
            self.frame_count = 0
            self.previous_detections: Optional[List[Detection]] = None
            
            # Range estimator for distance calculation
            try:
                from range_estimator import create_range_estimator
                self.range_estimator = create_range_estimator(max_distance=10.0)
                logging.info("Range estimator initialized")
            except Exception:
                logging.warning("Range estimator not available, using simple depth conversion")
                self.range_estimator = None

            logging.info("Model loaded successfully")
            logging.info(f"Available classes: {len(self.model.names)}")
        except Exception:
            logging.exception("Error initializing detection engine")
            raise

    def process_detections(self, results) -> List[Detection]:
        """
        Process YOLO results into a list of Detection objects.
        
        Args:
            results: The detection results returned by the YOLO model.
        
        Returns:
            A list of Detection objects.
        """
        detections: List[Detection] = []
        boxes = results.boxes

        for box in boxes:
            try:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Get confidence and class information
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Get tracking ID if available (YOLO tracking)
                tracking_id = None
                if hasattr(box, 'id') and box.id is not None and len(box.id) > 0:
                    tracking_id = int(box.id[0])

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    tracking_id=tracking_id
                )

                # Calculate movement if previous detections exist
                if self.previous_detections:
                    detection.movement = self.calculate_movement(detection, self.previous_detections)

                detections.append(detection)
            except Exception:
                logging.exception("Error processing a detection")
                continue

        return detections

    def calculate_movement(self, current_detection: Detection, previous_detections: List[Detection]) -> Optional[Dict]:
        """
        Calculate movement information for a given detection by comparing it with previous detections.
        
        Args:
            current_detection: The detection for which to calculate movement.
            previous_detections: A list of Detection objects from the previous frame.
        
        Returns:
            A dictionary with movement data (speed, direction, dx, dy) or None if no valid match is found.
        """
        try:
            best_iou = 0.0
            best_prev: Optional[Detection] = None

            for prev_detection in previous_detections:
                if prev_detection.class_id == current_detection.class_id:
                    iou = self.calculate_iou(current_detection.bbox, prev_detection.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev = prev_detection

            if best_prev and best_iou > 0.5:
                curr_center = (
                    (current_detection.bbox[0] + current_detection.bbox[2]) / 2,
                    (current_detection.bbox[1] + current_detection.bbox[3]) / 2
                )
                prev_center = (
                    (best_prev.bbox[0] + best_prev.bbox[2]) / 2,
                    (best_prev.bbox[1] + best_prev.bbox[3]) / 2
                )
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                speed = float(np.hypot(dx, dy))
                return {
                    'speed': speed,
                    'direction': self.get_direction(dx, dy),
                    'dx': float(dx),
                    'dy': float(dy)
                }
        except Exception:
            logging.exception("Error calculating movement")
        return None

    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: A tuple (x1, y1, x2, y2) for the first box.
            box2: A tuple (x1, y1, x2, y2) for the second box.
        
        Returns:
            The IoU as a float.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def get_direction(self, dx: float, dy: float) -> str:
        """
        Get the movement direction based on displacement (dx, dy).
        
        Args:
            dx: Displacement in x.
            dy: Displacement in y.
        
        Returns:
            A string arrow indicating the movement direction.
        """
        angle = np.degrees(np.arctan2(dy, dx))
        if -22.5 <= angle <= 22.5:
            return "R"  # right
        elif 22.5 < angle <= 67.5:
            return "DR"  # down-right
        elif 67.5 < angle <= 112.5:
            return "D"  # down
        elif 112.5 < angle <= 157.5:
            return "DL"  # down-left
        elif angle > 157.5 or angle <= -157.5:
            return "L"  # left
        elif -157.5 < angle <= -112.5:
            return "UL"  # up-left
        elif -112.5 < angle <= -67.5:
            return "U"  # up
        else:
            return "UR"  # up-right

    def draw_detection(self, frame: np.ndarray, detection: Detection, 
                       depth_map_raw: Optional[np.ndarray] = None) -> None:
        """
        Draw a single detection with distance (if depth available) or confidence.
        
        Args:
            frame: The frame on which to draw.
            detection: The Detection object.
            depth_map_raw: Optional normalized depth map (0-1 float32, higher=closer)
        """
        try:
            if not validate_frame(frame):
                return
            
            x1, y1, x2, y2 = detection.bbox
            
            # Validate bbox coordinates
            if x1 >= x2 or y1 >= y2:
                return
            if x1 < 0 or y1 < 0:
                return
            
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            if x1 >= x2 or y1 >= y2:
                return
            
            color = config.get_color(detection.class_name)
            
            # Draw border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Ensure class name is ASCII compatible
            class_name = detection.class_name
            if any(ord(char) > 127 for char in class_name):
                class_name = class_name.encode('ascii', 'replace').decode('ascii')
            
            # Calculate distance if depth map available
            distance_m = None
            if depth_map_raw is not None:
                try:
                    dh, dw = depth_map_raw.shape[:2]
                    # Scale bbox to depth map size if different
                    scale_x = dw / w
                    scale_y = dh / h
                    dx1 = int(x1 * scale_x)
                    dy1 = int(y1 * scale_y)
                    dx2 = int(x2 * scale_x)
                    dy2 = int(y2 * scale_y)
                    dx1 = max(0, min(dx1, dw - 1))
                    dy1 = max(0, min(dy1, dh - 1))
                    dx2 = max(0, min(dx2, dw))
                    dy2 = max(0, min(dy2, dh))
                    
                    if dx2 > dx1 and dy2 > dy1:
                        depth_region = depth_map_raw[dy1:dy2, dx1:dx2]
                        if depth_region.size > 0:
                            avg_depth = float(np.mean(depth_region))
                            
                            # Use range estimator if available (scientific method with tracking)
                            if self.range_estimator is not None:
                                # Get scene statistics for calibration
                                scene_min = float(np.min(depth_map_raw))
                                scene_max = float(np.max(depth_map_raw))
                                
                                # Estimate distance using scientific methods
                                range_est = self.range_estimator.estimate_distance(
                                    tracking_id=detection.tracking_id,
                                    normalized_depth=avg_depth,
                                    bbox=detection.bbox,
                                    object_class=class_name,
                                    frame_height=h,
                                    frame_width=w,
                                    scene_min=scene_min,
                                    scene_max=scene_max
                                )
                                distance_m = range_est.distance_m
                            else:
                                # Fallback: simple inverse relationship
                                # Scientific: distance ∝ 1 / depth
                                distance_m = 0.3 + (10.0 - 0.3) / (avg_depth + 0.1)
                                distance_m = max(0.3, min(12.0, distance_m))
                except Exception:
                    distance_m = None
            
            # Build label: show distance if available, else confidence
            if distance_m is not None:
                label = f"{class_name} {distance_m:.1f}m"
            else:
                conf_pct = int(detection.confidence * 100)
                label = f"{class_name} {conf_pct}%"
            
            # Add movement if available
            if detection.movement:
                speed = detection.movement.get('speed', 0)
                direction = detection.movement.get('direction', '')
                if speed > 0:
                    label += f" {direction}"
            
            # Draw label
            font_scale = 0.5
            font_thickness = 1
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            label_x = x1
            label_y = max(y1 - 6, label_size[1] + 10)
            pad_x = 6
            pad_y = 4
            
            label_bg_x1 = max(0, label_x - pad_x)
            label_bg_y1 = max(0, label_y - label_size[1] - pad_y)
            label_bg_x2 = min(w, label_x + label_size[0] + pad_x)
            label_bg_y2 = min(h, label_y + pad_y)
            
            # Draw label background
            cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (40, 40, 40), -1)
            cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x1 + 3, label_bg_y2), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        except Exception:
            logging.exception("Error drawing detection")

    def detect_and_track(self, frame: np.ndarray, 
                         depth_map_raw: Optional[np.ndarray] = None) -> Tuple[List[Detection], np.ndarray]:
        """
        Perform object detection on a frame with YOLO tracking enabled.
        
        Args:
            frame: The input frame.
            depth_map_raw: Optional normalized depth map for distance display (0-1 float32)
        
        Returns:
            A tuple containing a list of Detection objects and the annotated frame.
        """
        if not validate_frame(frame):
            return [], frame
        
        self.frame_count += 1
        annotated_frame = frame.copy()
        try:
            device = getattr(self, 'device', get_config_value(None, None, 'DEVICE', 'cpu'))
            # Enable tracking for smoother distance estimates
            results = self.model.track(
                frame,
                conf=get_config_value(None, None, 'CONFIDENCE_THRESHOLD', 0.5),
                iou=get_config_value(None, None, 'IOU_THRESHOLD', 0.5),
                imgsz=get_config_value(None, None, 'YOLO_IMAGE_SIZE', 640),
                device=device,
                persist=True,  # Maintain tracking across frames
                verbose=False  # Reduce logging for performance
            )[0]
            detections = self.process_detections(results)
            
            # Cleanup old trackers if range estimator is used
            if self.range_estimator is not None:
                active_ids = {d.tracking_id for d in detections if d.tracking_id is not None}
                self.range_estimator.cleanup_old_trackers(active_ids)
            
            for detection in detections:
                self.draw_detection(annotated_frame, detection, depth_map_raw)
            self.previous_detections = detections
            return detections, annotated_frame
        except Exception:
            logging.exception("Error in detect_and_track")
            return [], frame

    def add_safety_overlay(self, frame: np.ndarray, safety_assessment: Dict) -> np.ndarray:
        """
        Add a safety assessment overlay to the frame.
        
        Args:
            frame: The input frame.
            safety_assessment: A dictionary with safety assessment information.
        
        Returns:
            The frame with the safety overlay added.
        """
        try:
            if not safety_assessment:
                return frame

            danger_score = safety_assessment.get('danger_score', 0)
            reason = safety_assessment.get('reason', 'No hazards detected')
            navigation = safety_assessment.get('navigation', 'Path clear')
            alerts = safety_assessment.get('priority_alerts', [])
            color = (0, int(255 * (1 - danger_score)), int(255 * danger_score))
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (0, frame.shape[0] - 120),
                (frame.shape[1], frame.shape[0]),
                (0, 0, 0),
                -1
            )
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            y_pos = frame.shape[0] - 90
            cv2.putText(
                frame,
                f"Danger Score: {danger_score:.2f} | {reason}",
                (10, y_pos),
                config.FONT,
                config.FONT_SCALE * 1.2,
                color,
                config.FONT_THICKNESS
            )
            cv2.putText(
                frame,
                f"Navigation: {navigation}",
                (10, y_pos + 30),
                config.FONT,
                config.FONT_SCALE * 1.2,
                color,
                config.FONT_THICKNESS
            )
            if alerts:
                alert_text = f"Alerts: {', '.join(alerts)}"
                cv2.putText(
                    frame,
                    alert_text,
                    (10, y_pos + 60),
                    config.FONT,
                    config.FONT_SCALE,
                    color,
                    config.FONT_THICKNESS
                )
            return frame
        except Exception:
            logging.exception("Error adding safety overlay")
            return frame
#!/usr/bin/env python3
"""
ONNX Runtime DirectML-based Detection Engine for RTX 5070 Ti GPU
This uses DirectML which supports all modern GPUs including RTX 5070 Ti
"""
import cv2
import numpy as np
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime-directml not installed. Install with: pip install onnxruntime-directml")

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from config import config
import logging

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


class DetectionEngineONNX:
    """ONNX Runtime DirectML-based detection engine for GPU acceleration."""
    
    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize the detection engine with ONNX model.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'dml' for DirectML, or None for auto-detection)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime-directml is required. Install with: pip install onnxruntime-directml")
        
        try:
            # Determine device and provider
            if device is None:
                use_gpu = getattr(config, 'USE_GPU', False)
                if use_gpu:
                    # Try DirectML first (for Windows GPU including RTX 5070 Ti)
                    try:
                        # DirectML provider should be available in onnxruntime-directml
                        device = 'dml'
                        self.provider = 'DmlExecutionProvider'
                        logging.info("Using DirectML GPU provider for RTX 5070 Ti")
                    except Exception:
                        # Fallback to CPU
                        device = 'cpu'
                        self.provider = 'CPUExecutionProvider'
                        logging.info("DirectML not available, using CPU")
                else:
                    device = 'cpu'
                    self.provider = 'CPUExecutionProvider'
            
            # Load ONNX model
            import os
            from pathlib import Path
            
            # Get weight file name and convert to ONNX
            weight_file = getattr(config, 'WEIGHT_FILE', 'yolo12n.pt')
            onnx_file = weight_file.replace('.pt', '.onnx')
            
            # Try multiple locations
            weights_dir = getattr(config, 'WEIGHT_DIR', None) or Path('weights')
            possible_paths = [
                Path(onnx_file),  # Root directory
                Path(weights_dir) / onnx_file,  # Weights directory
                Path(weights_dir) / weight_file.replace('.pt', '.onnx'),  # Alternative
            ]
            
            full_onnx_path = None
            for path in possible_paths:
                if path.exists():
                    full_onnx_path = str(path)
                    break
            
            if not full_onnx_path:
                raise FileNotFoundError(
                    f"ONNX model not found. Tried: {[str(p) for p in possible_paths]}\n"
                    f"Please export YOLOv12 model: python -c \"from ultralytics import YOLO; YOLO('yolo12n.pt').export(format='onnx', imgsz=512)\""
                )
            
            logging.info(f"Loading ONNX model from: {full_onnx_path}")
            logging.info(f"Using provider: {self.provider}")
            
            # Create ONNX Runtime session
            # DirectML should be available by default in onnxruntime-directml
            if self.provider == 'DmlExecutionProvider':
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']  # Fallback to CPU
            else:
                providers = [self.provider]
            
            try:
                self.session = ort.InferenceSession(
                    full_onnx_path,
                    providers=providers
                )
                logging.info(f"Successfully created ONNX session with provider: {self.provider}")
            except Exception as e:
                logging.warning(f"Failed to create session with {self.provider}: {e}")
                # Fallback to CPU
                logging.info("Falling back to CPU provider")
                self.provider = 'CPUExecutionProvider'
                self.session = ort.InferenceSession(
                    full_onnx_path,
                    providers=['CPUExecutionProvider']
                )
                device = 'cpu'
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Get model metadata
            input_shape = self.session.get_inputs()[0].shape
            logging.info(f"Model input shape: {input_shape}")
            logging.info(f"Model output names: {self.output_names}")
            
            self.device = device
            self.frame_count = 0
            self.previous_detections: Optional[List[Detection]] = None
            
            # YOLO class names (standard COCO classes)
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]
            
            # Get image size from config
            self.imgsz = getattr(config, 'YOLO_IMAGE_SIZE', 640)
            self.conf_threshold = getattr(config, 'CONFIDENCE_THRESHOLD', 0.25)
            self.iou_threshold = getattr(config, 'IOU_THRESHOLD', 0.45)
            
            logging.info("ONNX model loaded successfully")
            logging.info(f"Using device: {device} with provider: {self.provider}")
            logging.info(f"Image size: {self.imgsz}, Confidence: {self.conf_threshold}, IOU: {self.iou_threshold}")
            
        except Exception:
            logging.exception("Error initializing ONNX detection engine")
            raise

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess frame for YOLO inference."""
        # Get original dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Handle both int and tuple image sizes
        if isinstance(self.imgsz, tuple):
            # Tuple format: (height, width)
            target_h, target_w = self.imgsz
            # Calculate scale to fit while maintaining aspect ratio
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            scale = min(scale_w, scale_h)  # Use min to fit within target dimensions
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target dimensions
            pad_h = (target_h - new_h) // 2
            pad_w = (target_w - new_w) // 2
            
            padded = cv2.copyMakeBorder(
                resized, pad_h, target_h - new_h - pad_h,
                pad_w, target_w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        else:
            # Integer format: use as short side (original behavior)
            scale = self.imgsz / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square
            pad_h = (self.imgsz - new_h) // 2
            pad_w = (self.imgsz - new_w) // 2
            
            padded = cv2.copyMakeBorder(
                resized, pad_h, self.imgsz - new_h - pad_h,
                pad_w, self.imgsz - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]
        
        return input_tensor, scale, (pad_w, pad_h)

    def postprocess(self, outputs: np.ndarray, scale: float, pad: Tuple[int, int],
                    orig_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess YOLO outputs to get detections."""
        detections: List[Detection] = []
        
        # YOLO output format: [batch, 84, 8400] or [batch, num_classes+4, num_boxes]
        # Format: [x_center, y_center, width, height, class_scores...]
        
        output = outputs[0]  # Remove batch dimension
        
        # Transpose: [8400, 84]
        if len(output.shape) == 3:
            output = output.transpose(1, 2, 0).reshape(-1, output.shape[1])
        elif len(output.shape) == 2:
            output = output.transpose(1, 0)
        
        # Extract boxes and scores
        boxes = output[:, :4]  # [x_center, y_center, width, height]
        scores = output[:, 4:]  # Class scores
        
        # Get class IDs and confidences
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        valid_mask = confidences >= self.conf_threshold
        boxes = boxes[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            return detections
        
        # Convert from center format to corner format
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Apply scale and padding
        x_center = (x_center - pad[0]) / scale
        y_center = (y_center - pad[1]) / scale
        w = w / scale
        h = h / scale
        
        # Convert to corner format
        x1 = (x_center - w / 2).astype(int)
        y1 = (y_center - h / 2).astype(int)
        x2 = (x_center + w / 2).astype(int)
        y2 = (y_center + h / 2).astype(int)
        
        # Clip to image bounds
        orig_h, orig_w = orig_shape[:2]
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in range(len(boxes))],
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return detections
        
        # Create Detection objects
        for idx in indices.flatten():
            class_id = int(class_ids[idx])
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            detection = Detection(
                bbox=(int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                confidence=float(confidences[idx]),
                class_id=class_id,
                class_name=class_name
            )
            detections.append(detection)
        
        return detections

    def calculate_movement(self, current_detection: Detection, previous_detections: List[Detection]) -> Optional[Dict]:
        """Calculate movement information for a given detection."""
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
        """Calculate Intersection over Union."""
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
        """Get movement direction."""
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

    def draw_detection(self, frame: np.ndarray, detection: Detection) -> None:
        """Draw detection on frame."""
        try:
            x1, y1, x2, y2 = detection.bbox
            color = config.get_color(detection.class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.LINE_THICKNESS)
            
            label_parts = [f"{detection.class_name} ({detection.confidence:.2f})"]
            if detection.movement:
                speed = detection.movement.get('speed', 0)
                direction = detection.movement.get('direction', '')
                if speed > 0:
                    label_parts.append(f"{speed:.1f}{direction}")
            label = " | ".join(label_parts)
            
            label_size = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                config.FONT,
                config.FONT_SCALE,
                config.COLORS.get('text', (255, 255, 255)),
                config.FONT_THICKNESS
            )
        except Exception:
            logging.exception("Error drawing detection")

    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Perform object detection on a frame using ONNX Runtime."""
        self.frame_count += 1
        annotated_frame = frame.copy()
        
        try:
            # Preprocess
            input_tensor, scale, pad = self.preprocess(frame)
            
            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            
            # Postprocess
            detections = self.postprocess(outputs[0], scale, pad, frame.shape)
            
            # Calculate movement
            for detection in detections:
                if self.previous_detections:
                    detection.movement = self.calculate_movement(detection, self.previous_detections)
            
            # Draw detections
            for detection in detections:
                self.draw_detection(annotated_frame, detection)
            
            self.previous_detections = detections
            return detections, annotated_frame
            
        except Exception:
            logging.exception("Error in detect_and_track")
            return [], frame


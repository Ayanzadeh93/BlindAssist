#!/usr/bin/env python3
"""Scientific range estimation module combining depth map and object detection.

Uses proper mathematical models for distance estimation:
1. Depth-based estimation (inverse relationship)
2. Size-based estimation (using bounding box and known object sizes)
3. Kalman filtering for tracking and smoothing
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass


@dataclass
class RangeEstimate:
    """Range estimate with confidence."""
    distance_m: float
    confidence: float
    method: str  # 'depth', 'size', 'fused'
    depth_value: Optional[float] = None
    bbox_area: Optional[float] = None


class KalmanFilter1D:
    """Simple 1D Kalman filter for distance tracking."""
    
    def __init__(self, initial_distance: float = 2.0, process_noise: float = 0.1, measurement_noise: float = 0.5):
        """
        Initialize Kalman filter.
        
        Args:
            initial_distance: Initial distance estimate (meters)
            process_noise: Process noise variance (how much distance can change)
            measurement_noise: Measurement noise variance (uncertainty in measurements)
        """
        self.distance = initial_distance
        self.uncertainty = 1.0  # Initial uncertainty
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
    
    def predict(self) -> float:
        """Predict next state (distance doesn't change much between frames)."""
        # Distance prediction: assume small change
        self.uncertainty += self.process_noise
        return self.distance
    
    def update(self, measurement: float, measurement_confidence: float = 1.0) -> float:
        """
        Update with new measurement.
        
        Args:
            measurement: New distance measurement
            measurement_confidence: Confidence in measurement (0-1)
        """
        # Adjust measurement noise based on confidence
        effective_noise = self.measurement_noise / max(0.1, measurement_confidence)
        
        # Kalman gain
        gain = self.uncertainty / (self.uncertainty + effective_noise)
        
        # Update estimate
        self.distance = self.distance + gain * (measurement - self.distance)
        self.uncertainty = (1 - gain) * self.uncertainty
        
        return self.distance
    
    def get_estimate(self) -> Tuple[float, float]:
        """Get current estimate and uncertainty."""
        return self.distance, self.uncertainty


class RangeEstimator:
    """Scientific range estimation using depth and object size."""
    
    # Typical object sizes in meters (for size-based estimation)
    OBJECT_SIZES = {
        'person': 1.7,  # Average height
        'chair': 0.9,
        'table': 0.75,
        'cup': 0.1,
        'bottle': 0.25,
        'laptop': 0.35,
        'mouse': 0.1,
        'keyboard': 0.4,
        'book': 0.25,
        'cell phone': 0.15,
        'tv': 0.6,
        'monitor': 0.4,
        'couch': 0.9,
        'bed': 0.5,
    }
    
    # Camera parameters (typical webcam)
    FOCAL_LENGTH_PIXELS = 600.0  # Approximate focal length in pixels
    SENSOR_WIDTH_MM = 3.68  # Typical webcam sensor width in mm
    
    def __init__(self, max_distance: float = 10.0):
        """
        Initialize range estimator.
        
        Args:
            max_distance: Maximum expected distance in meters
        """
        self.max_distance = max_distance
        self.trackers: Dict[int, KalmanFilter1D] = {}  # tracking_id -> KalmanFilter
        self.distance_history: Dict[int, deque] = {}  # tracking_id -> distance history
        self.max_history = 10
        
        logging.info(f"RangeEstimator initialized (max_distance={max_distance}m)")
    
    def depth_to_distance_scientific(self, normalized_depth: float, scene_min: float = 0.0, 
                                     scene_max: float = 1.0) -> float:
        """
        Convert normalized depth to distance using inverse relationship.
        
        Scientific basis: Depth perception follows inverse relationship.
        For monocular depth estimation: distance ∝ 1 / depth_value
        
        Args:
            normalized_depth: Normalized depth (0-1, where 1=closest)
            scene_min: Minimum depth in scene (for calibration)
            scene_max: Maximum depth in scene (for calibration)
        
        Returns:
            Distance in meters
        """
        if scene_max <= scene_min:
            # Fallback: use simple inverse
            distance = self.max_distance / (normalized_depth + 0.1)
            return max(0.2, min(self.max_distance, distance))
        
        # Normalize to scene-relative depth
        scene_relative = (normalized_depth - scene_min) / (scene_max - scene_min + 1e-6)
        scene_relative = max(0.0, min(1.0, scene_relative))
        
        # Inverse relationship: distance = a / (depth^b + c)
        # Parameters calibrated for typical indoor scenes
        a = 0.3  # Minimum distance (meters)
        b = 1.2  # Power factor (controls curve steepness)
        c = 0.05  # Small offset to prevent division by zero
        
        # Inverse power law
        distance = a + (self.max_distance - a) / ((scene_relative ** b) + c)
        
        return max(0.2, min(self.max_distance + 2.0, distance))
    
    def size_to_distance(self, bbox_area_pixels: float, object_class: str, 
                        frame_height: int, frame_width: int) -> Optional[float]:
        """
        Estimate distance from bounding box size (using known object sizes).
        
        Uses pinhole camera model: distance = (focal_length * real_size) / pixel_size
        
        Args:
            bbox_area_pixels: Bounding box area in pixels
            object_class: Object class name
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
        
        Returns:
            Estimated distance in meters, or None if object size unknown
        """
        if object_class not in self.OBJECT_SIZES:
            return None
        
        real_size_m = self.OBJECT_SIZES[object_class]
        
        # Estimate pixel size from bounding box area
        # Assume object is roughly square in image
        pixel_size = np.sqrt(bbox_area_pixels)
        
        if pixel_size < 1.0:
            return None
        
        # Pinhole camera model: distance = (focal_length * real_size) / pixel_size
        # Adjust focal length based on frame size
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        focal_length = self.FOCAL_LENGTH_PIXELS * (frame_diagonal / 1000.0)  # Scale with frame size
        
        distance = (focal_length * real_size_m) / pixel_size
        
        return max(0.2, min(self.max_distance + 2.0, distance))
    
    def fuse_estimates(self, depth_estimate: Optional[float], size_estimate: Optional[float],
                      depth_confidence: float = 0.7, size_confidence: float = 0.5) -> RangeEstimate:
        """
        Fuse multiple distance estimates using weighted average.
        
        Args:
            depth_estimate: Distance from depth map
            size_estimate: Distance from bounding box size
            depth_confidence: Confidence in depth estimate (0-1)
            size_confidence: Confidence in size estimate (0-1)
        
        Returns:
            Fused range estimate
        """
        estimates = []
        confidences = []
        
        if depth_estimate is not None:
            estimates.append(depth_estimate)
            confidences.append(depth_confidence)
        
        if size_estimate is not None:
            estimates.append(size_estimate)
            confidences.append(size_confidence)
        
        if not estimates:
            # No estimates available
            return RangeEstimate(
                distance_m=2.0,
                confidence=0.0,
                method='none'
            )
        
        # Weighted average
        total_weight = sum(confidences)
        if total_weight < 0.1:
            # Low confidence, use simple average
            fused_distance = np.mean(estimates)
            fused_confidence = 0.3
        else:
            fused_distance = np.average(estimates, weights=confidences)
            fused_confidence = min(1.0, total_weight / len(estimates))
        
        method = 'fused' if len(estimates) > 1 else ('depth' if depth_estimate else 'size')
        
        return RangeEstimate(
            distance_m=fused_distance,
            confidence=fused_confidence,
            method=method,
            depth_value=depth_estimate,
            bbox_area=size_estimate
        )
    
    def estimate_distance(self, tracking_id: Optional[int], normalized_depth: float,
                         bbox: Tuple[int, int, int, int], object_class: str,
                         frame_height: int, frame_width: int,
                         scene_min: float = 0.0, scene_max: float = 1.0) -> RangeEstimate:
        """
        Estimate distance using multiple methods and tracking.
        
        Args:
            tracking_id: Object tracking ID (for smoothing)
            normalized_depth: Normalized depth value (0-1)
            bbox: Bounding box (x1, y1, x2, y2)
            object_class: Object class name
            frame_height: Frame height
            frame_width: Frame width
            scene_min: Scene minimum depth
            scene_max: Scene maximum depth
        
        Returns:
            RangeEstimate with smoothed distance
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Method 1: Depth-based estimation
        depth_distance = self.depth_to_distance_scientific(normalized_depth, scene_min, scene_max)
        depth_confidence = 0.7  # Moderate confidence in depth
        
        # Method 2: Size-based estimation
        size_distance = self.size_to_distance(bbox_area, object_class, frame_height, frame_width)
        size_confidence = 0.5 if size_distance else 0.0
        
        # Fuse estimates
        estimate = self.fuse_estimates(depth_distance, size_distance, depth_confidence, size_confidence)
        
        # Apply tracking/smoothing if tracking ID available
        if tracking_id is not None:
            if tracking_id not in self.trackers:
                # Initialize Kalman filter for this object
                self.trackers[tracking_id] = KalmanFilter1D(
                    initial_distance=estimate.distance_m,
                    process_noise=0.1,
                    measurement_noise=0.3
                )
                self.distance_history[tracking_id] = deque(maxlen=self.max_history)
            
            # Update Kalman filter
            tracker = self.trackers[tracking_id]
            tracker.predict()
            smoothed_distance = tracker.update(estimate.distance_m, estimate.confidence)
            
            # Store in history
            self.distance_history[tracking_id].append(smoothed_distance)
            
            # Use smoothed estimate
            estimate.distance_m = smoothed_distance
            estimate.method = f"{estimate.method}_tracked"
        else:
            # No tracking, use raw estimate
            pass
        
        return estimate
    
    def cleanup_old_trackers(self, active_tracking_ids: set):
        """Remove trackers for objects that are no longer detected."""
        inactive_ids = set(self.trackers.keys()) - active_tracking_ids
        for tid in inactive_ids:
            del self.trackers[tid]
            if tid in self.distance_history:
                del self.distance_history[tid]


def create_range_estimator(max_distance: float = 10.0) -> RangeEstimator:
    """Create a range estimator instance."""
    return RangeEstimator(max_distance=max_distance)


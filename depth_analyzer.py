#!/usr/bin/env python3
"""Depth analysis module for extracting real-world distance estimates from depth maps.

This module provides utilities to:
1. Extract depth values from depth estimation models (Depth Anything V2)
2. Convert normalized depth to approximate real-world distances
3. Analyze depth statistics for regions of interest (bounding boxes)
4. Generate depth-aware descriptions for navigation
"""
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class DepthStats:
    """Statistics for a depth region."""
    mean_depth: float  # Average depth (0-1 normalized, 1=closest)
    min_depth: float   # Minimum depth (closest point)
    max_depth: float   # Maximum depth (farthest point)
    center_depth: float  # Depth at center point
    mean_distance_m: float  # Average distance in meters
    min_distance_m: float   # Minimum distance in meters (closest point)
    max_distance_m: float   # Maximum distance in meters (farthest point)
    center_distance_m: float  # Distance at center in meters
    confidence: float  # Confidence score (0-1, based on depth variance)


class DepthAnalyzer:
    """Analyzes depth maps to extract real-world distance information."""
    
    def __init__(self, max_distance: float = 10.0, near_threshold: float = 2.0, debug: bool = False):
        """Initialize depth analyzer.
        
        Args:
            max_distance: Maximum expected distance in meters (default: 10m)
            near_threshold: Distance threshold for "near" objects in meters (default: 2m)
            debug: Enable debug logging for depth values
        """
        self.max_distance = max_distance
        self.near_threshold = near_threshold
        self.debug = debug
        logging.info(f"DepthAnalyzer initialized: max_dist={max_distance}m, near_thresh={near_threshold}m, debug={debug}")
    
    def normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-1 range where 1=closest, 0=farthest.
        
        Args:
            depth_map: Raw depth map (can be grayscale or colormap)
        
        Returns:
            Normalized depth map (0-1 float32)
        """
        try:
            # Convert to grayscale if colored
            if len(depth_map.shape) == 3:
                depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            else:
                depth_gray = depth_map
            
            # Normalize to 0-1 range
            depth_norm = depth_gray.astype(np.float32) / 255.0
            
            # In Depth Anything V2 colormap (INFERNO), brighter = closer
            # So higher values = closer = higher normalized depth
            return depth_norm
        except Exception:
            logging.exception("Error normalizing depth map")
            return np.zeros_like(depth_map, dtype=np.float32)
    
    def depth_to_distance(self, depth_normalized: float) -> float:
        """Convert normalized depth (0-1) to approximate distance in meters.
        
        Depth Anything V2 raw output and colormap behavior:
        - Higher raw values / brighter colormap = CLOSER = lower distance
        - Lower raw values / darker colormap = FARTHER = higher distance
        
        Simple linear formula: distance = max_dist * (1 - depth) + min_offset
        
        Args:
            depth_normalized: Normalized depth value (0-1, where 1=closest/brightest)
        
        Returns:
            Approximate distance in meters (0.3 to 10.0m range)
        """
        # Simple linear: 1.0 (closest) -> 0.3m, 0.0 (farthest) -> 10m
        min_dist = 0.3
        max_dist = self.max_distance
        distance = max_dist * (1.0 - depth_normalized) + min_dist
        return max(min_dist, min(max_dist + 2.0, distance))
    
    def analyze_region(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int],
                      frame_width: int) -> Optional[DepthStats]:
        """Analyze depth statistics for a bounding box region.
        
        Args:
            depth_map: Normalized depth map (0-1 float32)
            bbox: Bounding box (x1, y1, x2, y2) in original frame coordinates
            frame_width: Width of original frame (for coordinate scaling)
        
        Returns:
            DepthStats object or None if analysis fails
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Normalize depth map if needed
            if depth_map.dtype != np.float32:
                depth_map = self.normalize_depth_map(depth_map)
            
            depth_h, depth_w = depth_map.shape[:2]
            
            # Scale coordinates if depth map width differs from frame width
            if depth_w != frame_width:
                scale_x = depth_w / frame_width
                x1_scaled = int(x1 * scale_x)
                x2_scaled = int(x2 * scale_x)
            else:
                x1_scaled = x1
                x2_scaled = x2
            
            # Clamp to valid range
            x1_scaled = max(0, min(x1_scaled, depth_w - 1))
            y1 = max(0, min(y1, depth_h - 1))
            x2_scaled = max(0, min(x2_scaled, depth_w - 1))
            y2 = max(0, min(y2, depth_h - 1))
            
            if x1_scaled >= x2_scaled or y1 >= y2:
                return None
            
            # Extract depth region
            depth_region = depth_map[y1:y2, x1_scaled:x2_scaled]
            
            if depth_region.size == 0:
                return None
            
            # Calculate statistics
            mean_depth = float(np.mean(depth_region))
            min_depth = float(np.min(depth_region))
            max_depth = float(np.max(depth_region))
            
            center_x = (x1_scaled + x2_scaled) // 2
            center_y = (y1 + y2) // 2
            center_depth = float(depth_map[center_y, center_x]) if (center_y < depth_h and center_x < depth_w) else mean_depth
            
            # Debug logging to see actual depth values
            if self.debug:
                logging.info(f"Depth region analysis: mean={mean_depth:.3f}, min={min_depth:.3f}, "
                           f"max={max_depth:.3f}, center={center_depth:.3f}")
            
            # Convert to distances
            mean_distance = self.depth_to_distance(mean_depth)
            min_distance = self.depth_to_distance(max_depth)  # Max depth = min distance
            max_distance = self.depth_to_distance(min_depth)  # Min depth = max distance
            center_distance = self.depth_to_distance(center_depth)
            
            if self.debug:
                logging.info(f"Distance conversion: mean={mean_distance:.1f}m, min={min_distance:.1f}m, "
                           f"max={max_distance:.1f}m, center={center_distance:.1f}m")
            
            # Calculate confidence based on depth variance (lower variance = higher confidence)
            depth_variance = float(np.var(depth_region))
            confidence = 1.0 - min(depth_variance * 4.0, 1.0)  # Scale variance to 0-1
            
            return DepthStats(
                mean_depth=mean_depth,
                min_depth=min_depth,
                max_depth=max_depth,
                center_depth=center_depth,
                mean_distance_m=round(mean_distance, 1),
                min_distance_m=round(min_distance, 1),
                max_distance_m=round(max_distance, 1),
                center_distance_m=round(center_distance, 1),
                confidence=round(confidence, 2)
            )
        except Exception:
            logging.exception("Error analyzing depth region")
            return None
    
    def get_distance_category(self, distance_m: float) -> str:
        """Convert distance to descriptive category.
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            Category string
        """
        if distance_m < 0.5:
            return "immediate"
        elif distance_m < 1.0:
            return "very close"
        elif distance_m < 2.0:
            return "close"
        elif distance_m < 4.0:
            return "near"
        elif distance_m < 6.0:
            return "medium distance"
        elif distance_m < 10.0:
            return "far"
        else:
            return "very far"
    
    def get_safety_level(self, distance_m: float) -> Tuple[str, float]:
        """Determine safety level based on distance.
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            Tuple of (safety_level, danger_score)
        """
        if distance_m < 0.5:
            return "critical", 0.95
        elif distance_m < 1.0:
            return "dangerous", 0.85
        elif distance_m < 2.0:
            return "caution", 0.6
        elif distance_m < 4.0:
            return "safe", 0.2
        else:
            return "clear", 0.0
    
    def analyze_scene_depth(self, depth_map: np.ndarray) -> Dict[str, Any]:
        """Analyze overall scene depth statistics.
        
        Args:
            depth_map: Depth map (will be normalized if needed)
        
        Returns:
            Dictionary with scene depth statistics
        """
        try:
            # Normalize depth map
            depth_norm = self.normalize_depth_map(depth_map)
            
            # Calculate overall statistics
            mean_depth = float(np.mean(depth_norm))
            min_depth = float(np.min(depth_norm))
            max_depth = float(np.max(depth_norm))
            median_depth = float(np.median(depth_norm))
            
            # Convert to distances
            mean_distance = self.depth_to_distance(mean_depth)
            min_distance = self.depth_to_distance(max_depth)
            max_distance = self.depth_to_distance(min_depth)
            median_distance = self.depth_to_distance(median_depth)
            
            # Calculate depth distribution
            h, w = depth_norm.shape[:2]
            
            # Divide into thirds (left, center, right)
            third_w = w // 3
            left_region = depth_norm[:, :third_w]
            center_region = depth_norm[:, third_w:2*third_w]
            right_region = depth_norm[:, 2*third_w:]
            
            left_mean_dist = self.depth_to_distance(float(np.mean(left_region)))
            center_mean_dist = self.depth_to_distance(float(np.mean(center_region)))
            right_mean_dist = self.depth_to_distance(float(np.mean(right_region)))
            
            # Find closest region
            closest_region = min([
                ("left", left_mean_dist),
                ("center", center_mean_dist),
                ("right", right_mean_dist)
            ], key=lambda x: x[1])
            
            return {
                "mean_distance_m": round(mean_distance, 1),
                "min_distance_m": round(min_distance, 1),
                "max_distance_m": round(max_distance, 1),
                "median_distance_m": round(median_distance, 1),
                "left_distance_m": round(left_mean_dist, 1),
                "center_distance_m": round(center_mean_dist, 1),
                "right_distance_m": round(right_mean_dist, 1),
                "closest_region": closest_region[0],
                "closest_distance_m": round(closest_region[1], 1),
                "overall_category": self.get_distance_category(mean_distance),
                "safety_level": self.get_safety_level(min_distance)[0],
                "danger_score": self.get_safety_level(min_distance)[1]
            }
        except Exception:
            logging.exception("Error analyzing scene depth")
            return {
                "mean_distance_m": 5.0,
                "min_distance_m": 1.0,
                "max_distance_m": 10.0,
                "error": "Failed to analyze depth"
            }
    
    def format_depth_info_for_llm(self, detection_info: List[Dict[str, Any]], 
                                 scene_stats: Dict[str, Any]) -> str:
        """Format depth information for LLM prompt.
        
        Args:
            detection_info: List of detection dictionaries with depth stats
            scene_stats: Overall scene depth statistics
        
        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        
        # Scene overview
        lines.append("=== DEPTH ANALYSIS ===")
        lines.append(f"Scene Overview: {scene_stats.get('overall_category', 'unknown')}")
        lines.append(f"- Average distance: {scene_stats.get('mean_distance_m', 0):.1f}m")
        lines.append(f"- Closest point: {scene_stats.get('min_distance_m', 0):.1f}m ({scene_stats.get('closest_region', 'unknown')} region)")
        lines.append(f"- Safety level: {scene_stats.get('safety_level', 'unknown')}")
        lines.append("")
        
        # Regional analysis
        lines.append("Regional Distances:")
        lines.append(f"- Left (9 o'clock): {scene_stats.get('left_distance_m', 0):.1f}m")
        lines.append(f"- Center (12 o'clock): {scene_stats.get('center_distance_m', 0):.1f}m")
        lines.append(f"- Right (3 o'clock): {scene_stats.get('right_distance_m', 0):.1f}m")
        lines.append("")
        
        # Object-specific depth
        if detection_info:
            lines.append("Detected Objects with Distances:")
            for info in detection_info[:8]:
                lines.append(
                    f"- {info['class_name']}: {info['clock_direction']} at {info['distance_avg']:.1f}m "
                    f"({info['distance_category']}, confidence: {info.get('depth_confidence', 0.8):.0%})"
                )
        else:
            lines.append("No objects detected")
        
        return "\n".join(lines)


def create_depth_analyzer(max_distance: float = 10.0, debug: bool = True) -> DepthAnalyzer:
    """Create a depth analyzer instance.
    
    Args:
        max_distance: Maximum expected distance in meters
        debug: Enable debug logging to see actual depth values
    
    Returns:
        DepthAnalyzer instance
    """
    return DepthAnalyzer(max_distance=max_distance, debug=debug)


def calibrate_depth_to_distance(depth_normalized: float, known_distance_m: float) -> None:
    """Helper function to calibrate depth-to-distance conversion.
    
    Use this to log depth values at known distances for calibration.
    
    Args:
        depth_normalized: Normalized depth value from depth map (0-1)
        known_distance_m: Your actual measured distance in meters
    
    Example:
        # Stand 1 meter from camera, check logs for depth value
        calibrate_depth_to_distance(0.75, 1.0)
        # This helps you adjust the depth_to_distance() formula
    """
    logging.info(f"CALIBRATION: depth_normalized={depth_normalized:.3f} → actual_distance={known_distance_m:.1f}m")
    logging.info(f"  Brightness: {depth_normalized*100:.0f}% (100%=brightest/closest, 0%=darkest/farthest)")
    
    # Suggest calibration adjustments
    if depth_normalized > 0.8 and known_distance_m > 2.0:
        logging.warning(f"  ⚠ Bright depth ({depth_normalized:.2f}) but far distance ({known_distance_m}m) - may need recalibration")
    elif depth_normalized < 0.4 and known_distance_m < 2.0:
        logging.warning(f"  ⚠ Dark depth ({depth_normalized:.2f}) but close distance ({known_distance_m}m) - may need recalibration")


#!/usr/bin/env python3
"""ArUco marker detection module for 3-digit markers (000-999)."""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from app_utils import validate_frame

@dataclass
class ArucoDetection:
    """ArUco marker detection result."""
    marker_id: int  # 0-999
    corners: np.ndarray  # 4 corner points
    center: Tuple[int, int]  # Center point (x, y)
    distance: float  # Estimated distance in meters
    confidence: float  # Detection confidence (0-1)


class ArucoDetector:
    """Detects ArUco markers with 3-digit IDs (000-999)."""
    
    # Common dictionaries that support 1000 markers
    SUPPORTED_DICTIONARIES = [
        'DICT_4X4_1000',
        'DICT_5X5_1000', 
        'DICT_6X6_1000',
        'DICT_7X7_1000',
        'DICT_ARUCO_ORIGINAL',  # Original ArUco (1024 markers)
    ]
    
    def __init__(self, 
                 marker_size_m: float = 0.15,  # 15cm for A4 paper
                 dictionary_type: Optional[str] = None,  # None = auto-detect
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 auto_detect_dictionary: bool = True):
        """
        Initialize ArUco detector with optimized parameters for accuracy.
        
        Args:
            marker_size_m: Physical marker size in meters (15cm = 0.15m)
            dictionary_type: ArUco dictionary type (must support 1000 markers)
            camera_matrix: Optional camera calibration matrix (3x3 numpy array)
                          For maximum accuracy, calibrate your camera using cv2.calibrateCamera()
            dist_coeffs: Optional distortion coefficients (4x1 or 5x1 numpy array)
        
        Note:
            For best accuracy, provide camera_matrix and dist_coeffs from camera calibration.
            Without calibration, distance estimation uses approximate focal length.
            Calibration improves distance accuracy significantly (typically <5% error vs ~10-15%).
            
        Args:
            auto_detect_dictionary: If True and dictionary_type is None, automatically try
                                   multiple dictionaries to find the correct one.
        """
        self.marker_size = marker_size_m
        self.auto_detect = auto_detect_dictionary and dictionary_type is None
        self.dictionary_type = dictionary_type or 'DICT_4X4_1000'
        
        # Initialize with primary dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.dictionary_type, cv2.aruco.DICT_4X4_1000)
        )
        
        # For auto-detection, prepare all dictionaries
        if self.auto_detect:
            self.all_dictionaries = {}
            for dict_name in self.SUPPORTED_DICTIONARIES:
                try:
                    dict_attr = getattr(cv2.aruco, dict_name, None)
                    if dict_attr is not None:
                        self.all_dictionaries[dict_name] = cv2.aruco.getPredefinedDictionary(dict_attr)
                except Exception:
                    continue
            logging.info(f"Auto-detect mode: Will try {len(self.all_dictionaries)} dictionaries")
            self.dictionary_auto_detected = False  # Flag to only auto-detect once
            self.auto_detect_frames_checked = 0
            self.auto_detect_frame_skip = 30  # Only check every 30 frames for auto-detect (was 10)
        else:
            self.all_dictionaries = {}
            self.dictionary_auto_detected = True  # Skip auto-detect if disabled
        
        # Handle different OpenCV versions
        try:
            # OpenCV 4.7+ uses ArucoDetector
            self.parameters = cv2.aruco.DetectorParameters()
            # Optimize parameters for accuracy (but balance with performance)
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # Disable for speed (was SUBPIX)
            # self.parameters.cornerRefinementWinSize = 5  # Disabled for performance
            # self.parameters.cornerRefinementMaxIterations = 30  # Disabled for performance
            # self.parameters.cornerRefinementMinAccuracy = 0.01  # Disabled for performance
            self.parameters.adaptiveThreshWinSizeMin = 5  # Larger window = faster (was 3)
            self.parameters.adaptiveThreshWinSizeMax = 23
            self.parameters.adaptiveThreshWinSizeStep = 15  # Larger step = faster (was 10)
            self.parameters.minMarkerPerimeterRate = 0.03  # Minimum marker perimeter
            self.parameters.maxMarkerPerimeterRate = 4.0  # Maximum marker perimeter
            self.parameters.polygonalApproxAccuracyRate = 0.03  # Polygon approximation accuracy
            self.parameters.minOtsuStdDev = 5.0  # Minimum Otsu threshold std dev
            self.parameters.perspectiveRemovePixelPerCell = 4  # Perspective removal
            self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
            self.parameters.maxErroneousBitsInBorderRate = 0.35  # Error tolerance
            self.parameters.minDistanceToBorder = 3  # Minimum distance to border
            self.parameters.markerBorderBits = 1  # Border bits
            self.parameters.detectInvertedMarker = False  # Don't detect inverted markers
            
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            self.use_new_api = True
        except AttributeError:
            # Fallback for older OpenCV versions
            self.parameters = cv2.aruco.DetectorParameters_create()
            # Apply same optimizations (performance-focused)
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # Disable for speed
            # self.parameters.cornerRefinementWinSize = 5  # Disabled for performance
            # self.parameters.cornerRefinementMaxIterations = 30  # Disabled for performance
            # self.parameters.cornerRefinementMinAccuracy = 0.01  # Disabled for performance
            self.parameters.adaptiveThreshWinSizeMin = 5  # Larger window = faster
            self.parameters.adaptiveThreshWinSizeMax = 23
            self.parameters.adaptiveThreshWinSizeStep = 15  # Larger step = faster
            self.parameters.minMarkerPerimeterRate = 0.03
            self.parameters.maxMarkerPerimeterRate = 4.0
            self.parameters.polygonalApproxAccuracyRate = 0.03
            self.parameters.minOtsuStdDev = 5.0
            self.parameters.perspectiveRemovePixelPerCell = 4
            self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
            self.parameters.maxErroneousBitsInBorderRate = 0.35
            self.parameters.minDistanceToBorder = 3
            self.parameters.markerBorderBits = 1
            self.parameters.detectInvertedMarker = False
            
            self.detector = None  # Will use detectMarkers directly
            self.use_new_api = False
        
        # Camera calibration (optional, improves accuracy)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # Detection history for smoothing
        self.detection_history: Dict[int, List[float]] = {}
        self.history_size = 5
        
        logging.info(f"ArUco detector initialized: {self.dictionary_type}, marker size: {marker_size_m}m")
    
    def _detect_with_dictionary(self, gray: np.ndarray, dictionary) -> Tuple:
        """Detect markers with a specific dictionary."""
        if self.use_new_api:
            # Create temporary detector for this dictionary
            temp_detector = cv2.aruco.ArucoDetector(dictionary, self.parameters)
            corners, ids, rejected = temp_detector.detectMarkers(gray)
        else:
            # Fallback for older OpenCV versions
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=self.parameters
            )
        return corners, ids, rejected
    
    def detect(self, frame: np.ndarray) -> Tuple[List[ArucoDetection], np.ndarray]:
        """
        Detect ArUco markers in frame.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Tuple of (detections, annotated_frame)
        """
        if not validate_frame(frame):
            return [], frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optional: Apply histogram equalization for better contrast
        # This can improve detection in varying lighting conditions
        # gray = cv2.equalizeHist(gray)
        
        # Auto-detect dictionary if enabled (but only once, not every frame!)
        rejected = []
        if self.auto_detect and len(self.all_dictionaries) > 0 and not self.dictionary_auto_detected:
            # Only try auto-detect every N frames to avoid performance hit
            self.auto_detect_frames_checked += 1
            if self.auto_detect_frames_checked % self.auto_detect_frame_skip == 0:
                # Try primary dictionary first
                corners, ids, rejected = self._detect_with_dictionary(gray, self.dictionary)
                
                # If no markers found, try other dictionaries (but only once!)
                if ids is None or len(ids) == 0:
                    for dict_name, test_dict in self.all_dictionaries.items():
                        if dict_name == self.dictionary_type:
                            continue  # Already tried
                        
                        test_corners, test_ids, test_rejected = self._detect_with_dictionary(gray, test_dict)
                        if test_ids is not None and len(test_ids) > 0:
                            # Found markers with this dictionary!
                            marker_ids = test_ids.flatten().tolist()
                            logging.info(f"✓ Found markers using dictionary: {dict_name} (IDs: {marker_ids})")
                            logging.info(f"  Switched from {self.dictionary_type} to {dict_name}")
                            self.dictionary = test_dict
                            self.dictionary_type = dict_name
                            # Recreate detector with correct dictionary
                            if self.use_new_api:
                                self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
                            corners, ids, rejected = test_corners, test_ids, test_rejected
                            self.dictionary_auto_detected = True  # Mark as detected, stop trying
                            break
                else:
                    # Primary dictionary worked, mark as detected
                    self.dictionary_auto_detected = True
            else:
                # Skip auto-detect this frame, just use current dictionary
                corners, ids, rejected = self._detect_with_dictionary(gray, self.dictionary)
        else:
            # Use single dictionary (fast path - no auto-detect)
            corners, ids, rejected = self._detect_with_dictionary(gray, self.dictionary)
        
        # Validate detection results
        if ids is not None:
            # Filter out invalid detections
            valid_indices = []
            for i, marker_id in enumerate(ids.flatten()):
                if 0 <= marker_id <= 999:
                    # Check if corners are valid
                    if i < len(corners) and corners[i] is not None and len(corners[i]) > 0:
                        if corners[i][0].shape[0] == 4:  # Must have 4 corners
                            valid_indices.append(i)
            
            if len(valid_indices) < len(ids):
                # Filter to only valid detections
                invalid_count = len(ids) - len(valid_indices)
                if invalid_count > 0:
                    logging.debug(f"Filtered out {invalid_count} invalid marker detection(s)")
                ids = ids[valid_indices]
                corners = [corners[i] for i in valid_indices]
                if len(ids) == 0:
                    ids = None
        
        annotated = frame.copy()
        detections = []
        
        if ids is not None and len(ids) > 0:
            # Process each detected marker
            for i, marker_id in enumerate(ids.flatten()):
                # Validate marker ID range (0-999)
                if marker_id < 0 or marker_id > 999:
                    logging.warning(f"Invalid marker ID detected: {marker_id} (expected 0-999)")
                    continue
                
                marker_corners = corners[i][0]
                
                # Validate corner points (must have 4 corners)
                if marker_corners.shape[0] != 4:
                    logging.warning(f"Invalid marker corners: expected 4, got {marker_corners.shape[0]}")
                    continue
                
                # Validate corner points are within frame bounds
                h, w = frame.shape[:2]
                if np.any(marker_corners[:, 0] < 0) or np.any(marker_corners[:, 0] >= w) or \
                   np.any(marker_corners[:, 1] < 0) or np.any(marker_corners[:, 1] >= h):
                    logging.debug(f"Marker {marker_id} corners out of bounds, skipping")
                    continue
                
                # Calculate center point (more accurate using centroid)
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                center = (center_x, center_y)
                
                # Estimate distance using pose estimation (more accurate)
                distance = self._estimate_distance(marker_corners, frame.shape)
                
                # Update detection history for smoothing
                if marker_id not in self.detection_history:
                    self.detection_history[marker_id] = []
                self.detection_history[marker_id].append(distance)
                if len(self.detection_history[marker_id]) > self.history_size:
                    self.detection_history[marker_id].pop(0)
                
                # Use smoothed distance
                smoothed_distance = np.mean(self.detection_history[marker_id])
                
                detection = ArucoDetection(
                    marker_id=int(marker_id),
                    corners=marker_corners,
                    center=center,
                    distance=smoothed_distance,
                    confidence=1.0  # ArUco detection is binary (detected or not)
                )
                detections.append(detection)
                
                # Draw marker on frame
                self._draw_marker(annotated, detection)
        
        return detections, annotated
    
    def _estimate_distance(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Estimate distance to marker in meters using pose estimation for accuracy.
        
        Args:
            corners: Marker corner points (4 corners, shape: (4, 2))
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            Estimated distance in meters
        """
        try:
            # Use pose estimation for accurate distance calculation
            # This is more accurate than simple pixel size estimation
            
            # Create object points (marker corners in 3D space)
            # Marker is in XY plane, centered at origin
            marker_half = self.marker_size / 2.0
            obj_points = np.array([
                [-marker_half, marker_half, 0],   # Top-left
                [marker_half, marker_half, 0],    # Top-right
                [marker_half, -marker_half, 0],   # Bottom-right
                [-marker_half, -marker_half, 0]   # Bottom-left
            ], dtype=np.float32)
            
            # Reshape corners for solvePnP
            image_points = corners.reshape(-1, 1, 2).astype(np.float32)
            
            # If camera matrix is available, use it for accurate pose estimation
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                # Use calibrated camera parameters
                camera_matrix = self.camera_matrix
                dist_coeffs = self.dist_coeffs
            else:
                # Create approximate camera matrix from frame dimensions
                # This is less accurate but works without calibration
                h, w = frame_shape[:2]
                focal_length = max(w, h) * 0.8  # Approximate focal length
                cx, cy = w / 2.0, h / 2.0
                camera_matrix = np.array([
                    [focal_length, 0, cx],
                    [0, focal_length, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # No distortion
            
            # Solve PnP to get pose (rotation and translation vectors)
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Distance is the Z component of translation vector (in meters)
                # tvec[2] is the distance along the camera's Z-axis
                distance = float(np.linalg.norm(tvec))
                return max(0.1, min(distance, 10.0))  # Clamp between 0.1m and 10m
            else:
                # Fallback to pixel-based estimation if pose estimation fails
                return self._estimate_distance_fallback(corners, frame_shape)
                
        except Exception as e:
            logging.debug(f"Pose estimation failed: {e}, using fallback")
            return self._estimate_distance_fallback(corners, frame_shape)
    
    def _estimate_distance_fallback(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        """
        Fallback distance estimation using pixel size (less accurate).
        
        Args:
            corners: Marker corner points
            frame_shape: Frame dimensions (height, width)
        
        Returns:
            Estimated distance in meters
        """
        # Calculate marker size in pixels (average of width and height)
        width_px = np.linalg.norm(corners[0] - corners[1])
        height_px = np.linalg.norm(corners[1] - corners[2])
        marker_pixel_size = (width_px + height_px) / 2.0  # Average for better accuracy
        
        if marker_pixel_size <= 0:
            return 0.0
        
        # Estimate focal length from frame dimensions
        h, w = frame_shape[:2]
        frame_diagonal = np.sqrt(w**2 + h**2)
        focal_length = frame_diagonal * 0.8  # Approximate focal length
        
        # Pinhole camera model: distance = (real_size * focal_length) / pixel_size
        distance = (self.marker_size * focal_length) / marker_pixel_size
        
        return max(0.1, min(distance, 10.0))  # Clamp between 0.1m and 10m
    
    def _draw_marker(self, frame: np.ndarray, detection: ArucoDetection) -> None:
        """Draw detected marker on frame with ID and distance."""
        # Draw marker outline
        corners_int = detection.corners.astype(int)
        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(frame, detection.center, 5, (0, 255, 0), -1)
        
        # Format marker ID as 3-digit string (000-999)
        marker_id_str = f"{detection.marker_id:03d}"
        
        # Draw ID and distance
        label = f"ID: {marker_id_str} ({detection.distance:.2f}m)"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Position label above marker
        label_x = detection.center[0] - label_size[0] // 2
        label_y = detection.center[1] - 20
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (label_x - 5, label_y - label_size[1] - 5),
            (label_x + label_size[0] + 5, label_y + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    def reset_auto_detect(self) -> None:
        """Reset auto-detect flag to allow dictionary detection again."""
        if self.auto_detect:
            self.dictionary_auto_detected = False
            self.auto_detect_frames_checked = 0
            logging.info("ArUco auto-detect reset - will try dictionaries again")
    
    def get_detection_summary(self, detections: List[ArucoDetection]) -> str:
        """
        Get text summary of detections for audio/TTS.
        
        Args:
            detections: List of detected markers
        
        Returns:
            Summary string
        """
        if not detections:
            return "No markers detected"
        
        if len(detections) == 1:
            det = detections[0]
            return f"Marker {det.marker_id:03d} detected, {det.distance:.1f} meters away"
        else:
            ids = [f"{d.marker_id:03d}" for d in detections]
            return f"{len(detections)} markers detected: {', '.join(ids)}"


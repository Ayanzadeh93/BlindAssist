# Import necessary libraries
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

class VideoSource:
    """
    Handles video input from camera device, video file, or network stream.
    Provides unified interface for reading frames and managing video resources.
    
    Supports:
    - Local camera: int (camera index, e.g., 0, 1)
    - Video file: str/Path (file path)
    - Network stream: str (URL like http://IP:PORT/video, rtsp://IP:PORT/stream)
    
    For phone cameras, common apps include:
    - IP Webcam (Android): http://IP:8080/video
    - DroidCam (Android): http://IP:4747/mjpeg
    - EpocCam (iOS): http://IP:PORT/video
    - Sensor Pro (iOS): Check app documentation for stream URL
    """
    
    def __init__(self, source: Union[int, str, Path], output_path: Optional[Path] = None):
        """
        Initialize video source with camera index, video file path, or network stream URL.
        
        Args:
            source: Camera index (int), video file path (str/Path), or network stream URL (str)
            output_path: Optional path to save processed video
        """
        self.source = source
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self._setup_source()
    
    def _is_network_stream(self, source) -> bool:
        """Check if source is a network stream (URL)."""
        if isinstance(source, str):
            return source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://', 'tcp://', 'udp://'))
        return False
    
    def _find_working_camera(self, start_index: int = 0, max_cameras: int = 5) -> Optional[int]:
        """
        Automatically find a working camera by testing multiple camera indices.
        Prioritizes integrated laptop cameras.
        
        Args:
            start_index: Starting camera index to test
            max_cameras: Maximum number of cameras to test
            
        Returns:
            Working camera index, or None if no camera found
        """
        logging.info(f"Auto-detecting working camera (testing indices {start_index} to {start_index + max_cameras - 1})...")
        
        for camera_id in range(start_index, start_index + max_cameras):
            try:
                logging.info(f"Testing camera device {camera_id}...")
                test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
                
                if test_cap.isOpened():
                    # Try to read a frame to verify it actually works
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        # Get camera properties
                        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        test_cap.release()
                        logging.info(f"✅ Found working camera at device {camera_id} ({width}x{height})")
                        return camera_id
                    else:
                        logging.warning(f"Camera {camera_id} opened but cannot read frames")
                test_cap.release()
            except Exception as e:
                logging.debug(f"Camera {camera_id} test failed: {e}")
                continue
        
        logging.error("❌ No working camera found after testing all indices")
        return None
    
    def _test_camera_read(self, camera_id: int, timeout: float = 2.0) -> bool:
        """
        Test if a camera can actually read frames (not just open).
        
        Args:
            camera_id: Camera index to test
            timeout: Maximum time to wait for a frame
            
        Returns:
            True if camera can read frames, False otherwise
        """
        try:
            test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if not test_cap.isOpened():
                return False
            
            # Set a short timeout
            test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to read multiple frames to ensure it's working
            for _ in range(3):
                ret, frame = test_cap.read()
                if not ret or frame is None:
                    test_cap.release()
                    return False
            
            test_cap.release()
            return True
        except Exception:
            return False
    
    def _setup_source(self):
        """
        Set up video capture and get video properties.
        Handles both camera and file inputs with appropriate error checking.
        For network streams, adds timeout and retry logic.
        """
        try:
            is_network = self._is_network_stream(self.source)
            
            if is_network:
                # For network streams, configure with timeout
                logging.info(f"Connecting to network stream: {self.source}")
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                
                # Set buffer size to reduce latency for network streams
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Try to read a frame to verify connection
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    # Try alternative URL formats if first attempt fails
                    if isinstance(self.source, str) and ':' in self.source:
                        alternative_urls = self._get_alternative_urls(self.source)
                        for alt_url in alternative_urls:
                            logging.info(f"Trying alternative URL: {alt_url}")
                            if self.cap:
                                self.cap.release()
                            self.cap = cv2.VideoCapture(alt_url, cv2.CAP_FFMPEG)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                self.source = alt_url
                                logging.info(f"Successfully connected to: {alt_url}")
                                break
                        else:
                            raise RuntimeError(
                                f"Could not connect to network stream. Tried:\n"
                                f"  - {self.source}\n"
                                f"  - {chr(10).join('  - ' + url for url in alternative_urls)}\n"
                                f"Make sure your phone camera app is running and both devices are on the same network."
                            )
                    else:
                        raise RuntimeError(
                            f"Could not connect to network stream: {self.source}\n"
                            f"Make sure your phone camera app is running and both devices are on the same network."
                        )
            else:
                # For local camera/file
                if isinstance(self.source, int):
                    # Camera device - try to open with DirectShow backend on Windows
                    self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                    
                    # If camera fails to open, try auto-detection
                    if not self.cap.isOpened():
                        logging.warning(f"Camera {self.source} failed to open. Attempting auto-detection...")
                        if self.cap:
                            self.cap.release()
                        
                        # Try to find a working camera
                        working_camera = self._find_working_camera(start_index=0, max_cameras=5)
                        if working_camera is not None:
                            self.source = working_camera
                            self.cap = cv2.VideoCapture(working_camera, cv2.CAP_DSHOW)
                            logging.info(f"Using auto-detected camera: {working_camera}")
                        else:
                            raise RuntimeError(
                                f"Could not open camera {self.source} and no other working cameras found. "
                                f"Please check:\n"
                                f"  1. Camera is not being used by another application\n"
                                f"  2. Camera drivers are installed\n"
                                f"  3. Camera permissions are granted"
                            )
                else:
                    # Video file
                    self.cap = cv2.VideoCapture(str(self.source))
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.source}")
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if 0
            
            # For cameras, verify we can actually read frames
            if isinstance(self.source, int):
                # Test frame reading
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    logging.warning(f"Camera {self.source} opened but cannot read frames. Trying auto-detection...")
                    self.cap.release()
                    
                    # Try to find a working camera
                    working_camera = self._find_working_camera(start_index=0, max_cameras=5)
                    if working_camera is not None:
                        self.source = working_camera
                        self.cap = cv2.VideoCapture(working_camera, cv2.CAP_DSHOW)
                        # Update properties with new camera
                        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                        logging.info(f"Successfully switched to camera: {working_camera}")
                    else:
                        raise RuntimeError(
                            f"Camera {self.source} cannot read frames and no other working cameras found."
                        )
            
            # Log additional information
            if isinstance(self.source, (str, Path)) and not is_network:
                # Video file information
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    duration = total_frames / self.fps if self.fps > 0 else 0
                    logging.info(f"Loaded video file: {self.source}")
                    logging.info(f"Resolution: {self.frame_width}x{self.frame_height}")
                    logging.info(f"FPS: {self.fps}")
                    logging.info(f"Duration: {duration:.2f} seconds")
            elif is_network:
                logging.info(f"Network stream connected: {self.source}")
                logging.info(f"Resolution: {self.frame_width}x{self.frame_height}")
                logging.info(f"FPS: {self.fps}")
            
        except Exception as e:
            logging.error(f"Error setting up video source: {e}")
            raise
    
    def _get_alternative_urls(self, url: str) -> list:
        """
        Generate alternative URL formats to try for network streams.
        
        Args:
            url: Original URL
            
        Returns:
            List of alternative URLs to try
        """
        alternatives = []
        
        # Extract IP and port if present
        if '://' in url:
            protocol, rest = url.split('://', 1)
            if '/' in rest:
                host_port, path = rest.split('/', 1)
            else:
                host_port, path = rest, ''
        else:
            protocol = 'http'
            if '/' in url:
                host_port, path = url.split('/', 1)
            else:
                host_port, path = url, ''
        
        if ':' in host_port:
            ip, port = host_port.split(':', 1)
        else:
            ip = host_port
            port = '8080'  # Default port
        
        # Common paths for phone camera apps
        common_paths = ['/video', '/mjpeg', '/stream', '/live', '/videofeed', '']
        
        # Generate alternatives with different paths
        for alt_path in common_paths:
            alt_url = f"{protocol}://{ip}:{port}/{alt_path}".rstrip('/')
            if alt_url != url:
                alternatives.append(alt_url)
        
        # Also try different ports
        common_ports = ['8080', '4747', '8554', '1935', '5000']
        for alt_port in common_ports:
            if alt_port != port:
                for path in ['/video', '/mjpeg', '/stream']:
                    alt_url = f"{protocol}://{ip}:{alt_port}/{path}"
                    if alt_url != url:
                        alternatives.append(alt_url)
        
        return alternatives[:5]  # Limit to 5 alternatives to avoid too many attempts
    

    def get_output_fps(self) -> float:
        """
        Get appropriate output FPS for video writing.
        For video files, use original FPS. For camera, use 30 FPS.
        """
        if isinstance(self.source, (str, Path)):
            # For video files, use original FPS
            return self.fps
        else:
            # For camera input, use 30 FPS
            return 30.0
    
    def setup_writer(self, codec=cv2.VideoWriter_fourcc(*'mp4v')):
        """
        Set up video writer if output path is specified.
        Uses appropriate codec and frame rate from input source.
        """
        if self.output_path:
            try:
                # Use appropriate FPS for output
                output_fps = self.get_output_fps()
                
                self.writer = cv2.VideoWriter(
                    str(self.output_path),
                    codec,
                    output_fps,
                    (self.frame_width, self.frame_height)
                )
                logging.info(f"Video writer initialized: {self.output_path}")
                logging.info(f"Output FPS: {output_fps}")
            except Exception as e:
                logging.error(f"Error setting up video writer: {e}")
                raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple containing:
            - Boolean indicating if read was successful
            - Frame as numpy array (None if read failed)
        """
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to the output video if writer is initialized.
        
        Args:
            frame: Frame to write as numpy array
        """
        if self.writer is not None:
            self.writer.write(frame)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self.frame_width, self.frame_height
    
    def release(self):
        """Release video capture and writer resources."""
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
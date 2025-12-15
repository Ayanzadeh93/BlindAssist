#!/usr/bin/env python3
"""Metrics and logging module for AIDGPT application."""
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from gpt_interface import GPTResponse


@dataclass
class PerformanceMetrics:
    """Performance metrics tracker for the application."""
    fps: float = 0.0
    processing_time: float = 0.0
    frame_count: int = 0
    detection_count: int = 0
    gpt_calls: int = 0

    def update(self, frame_time: float, num_detections: int) -> None:
        """Update metrics with new frame data."""
        self.frame_count += 1
        self.processing_time += frame_time
        self.detection_count += num_detections
        self.fps = self.frame_count / max(self.processing_time, 0.001)


class CSVLogger:
    """CSV logger for GPT responses with video timestamps."""
    
    def __init__(self, csv_file: str, log_interval: float = 1.0):
        self.csv_file = csv_file
        self.log_interval = log_interval
        self.last_log_time = 0.0
        self.video_start_time: Optional[float] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_file_handle = None
        self.setup_csv()
    
    def setup_csv(self) -> None:
        """Setup CSV file with headers."""
        try:
            self.csv_file_handle = open(self.csv_file, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file_handle)
            # Write headers
            self.csv_writer.writerow([
                'video_time_seconds', 
                'timestamp', 
                'gpt_response', 
                'danger_score', 
                'model_used',
                'detection_count',
                'frame_number',
                'processing_time_ms'
            ])
            logging.info(f"CSV logging enabled: {self.csv_file}")
        except Exception as e:
            logging.error(f"Error setting up CSV logging: {e}")
            self.csv_writer = None
    
    def set_video_start_time(self, start_time: float) -> None:
        """Set the video start time for timestamp calculation."""
        self.video_start_time = start_time
    
    def log_response(self, gpt_response: GPTResponse, current_time: float, 
                    detection_count: int = 0, frame_number: int = 0, 
                    processing_time: float = 0.0) -> None:
        """Log GPT response to CSV if enough time has passed."""
        if not self.csv_writer:
            return
        
        # Check if enough time has passed since last log
        if current_time - self.last_log_time < self.log_interval:
            return
        
        try:
            # Calculate video time
            video_time = current_time - self.video_start_time if self.video_start_time else 0.0
            
            # Prepare response text
            response_text = gpt_response.reason
            if getattr(gpt_response, 'navigation', None):
                response_text += f" {gpt_response.navigation}"
            
            # Write to CSV
            self.csv_writer.writerow([
                f"{video_time:.2f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                response_text,
                gpt_response.danger_score,
                gpt_response.model_used,
                detection_count,
                frame_number,
                f"{processing_time*1000:.1f}"
            ])
            self.csv_file_handle.flush()  # Ensure data is written
            self.last_log_time = current_time
            
            logging.info(f"Logged to CSV: {video_time:.2f}s - {response_text[:50]}...")
            
        except Exception as e:
            logging.error(f"Error logging to CSV: {e}")
    
    def close(self) -> None:
        """Close CSV file."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            logging.info(f"CSV logging closed: {self.csv_file}")


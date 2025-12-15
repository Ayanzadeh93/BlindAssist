#!/usr/bin/env python3
"""Overlay rendering module for AIDGPT application."""
import cv2
import numpy as np
from typing import Dict, Tuple
from config import config
from application_state import ApplicationState


def _draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], 
               font_scale: float, color: Tuple[int, int, int], 
               thickness: int = 2) -> None:
    """Draw text on image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def _draw_panel(img: np.ndarray, x: int, y: int, width: int, height: int,
                bg_color: Tuple[int, int, int], border_color: Tuple[int, int, int]) -> None:
    """Draw a panel with background and border."""
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), border_color, 2)


def add_overlay(frame: np.ndarray, state: ApplicationState, frame_id: int) -> np.ndarray:
    """Add optimized debug information overlay to the frame.
    
    Args:
        frame: Input frame
        state: ApplicationState instance
        frame_id: Current frame number
        
    Returns:
        Frame with debug overlay
    """
    ui = config.UI_COLORS
    fonts = config.UI_FONTS
    metrics = state.metrics
    
    # Panel dimensions
    panel_x = 10
    panel_y = 10
    panel_width = 280
    panel_height = 130
    
    _draw_panel(frame, panel_x, panel_y, panel_width, panel_height,
                ui['bg_dark'], ui['primary'])
    
    # Header
    header_y = panel_y + 25
    _draw_text(frame, "DEBUG INFO", (panel_x + 15, header_y),
               fonts['heading_scale'], ui['primary'], fonts['heading_thickness'])
    
    # Info items
    info_y = header_y + 25
    info_items = [
        ("Frame:", f"{frame_id}", ui['text_primary']),
        ("Detect:", f"{metrics.detection_count}", ui['success'] if metrics.detection_count > 0 else ui['text_secondary']),
        ("FPS:", f"{metrics.fps:.1f}", ui['success'] if metrics.fps > 20 else ui['warning']),
        ("Status:", "Paused" if state.paused else "Run", ui['warning'] if state.paused else ui['success']),
    ]
    
    for label, value, color in info_items:
        _draw_text(frame, label, (panel_x + 15, info_y),
                   fonts['body_scale'], ui['text_secondary'], fonts['body_thickness'])
        
        value_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'], fonts['body_thickness'])[0]
        value_x = panel_x + panel_width - 15 - value_size[0]
        _draw_text(frame, value, (value_x, info_y),
                   fonts['body_scale'], color, fonts['body_thickness'])
        
        info_y += 23
    
    return frame


def add_controls_overlay(frame: np.ndarray, state: ApplicationState) -> np.ndarray:
    """Add optimized controls and status overlay to the frame.
    
    Args:
        frame: Input frame
        state: ApplicationState instance
        
    Returns:
        Frame with controls overlay
    """
    if not getattr(state, "show_ui_panel", False):
        return frame
    
    ui = config.UI_COLORS
    fonts = config.UI_FONTS
    h, w = frame.shape[:2]
    
    # === CONTROLS PANEL (Top Right) - Simplified ===
    controls_width = 220
    controls_items = [
        ("CONTROLS", None),
        ("Q:Quit", "P:Pause", "D:Debug"),
        ("M:Depth", "B:Detect", "G:GPT"),
        ("A:Audio", "T:Test", "K:Spatial"),
        ("H:Help", "U:UI", ""),
    ]
    
    controls_height = len(controls_items) * 22 + 30
    ctrl_x = w - controls_width - 10
    ctrl_y = 10
    
    _draw_panel(frame, ctrl_x, ctrl_y, controls_width, controls_height,
                ui['bg_dark'], ui['primary_dark'])
    
    # Draw controls
    y_offset = ctrl_y + 20
    for item in controls_items:
        if item[0] and item[1] is None:  # Header
            _draw_text(frame, item[0], (ctrl_x + 15, y_offset),
                       fonts['heading_scale'], ui['accent'], fonts['heading_thickness'])
        else:
            x_offset = ctrl_x + 15
            for text in item:
                if text:
                    _draw_text(frame, text, (x_offset, y_offset),
                               fonts['small_scale'], ui['text_secondary'], fonts['small_thickness'])
                    x_offset += 65
        y_offset += 22
    
    # === STATUS PANEL (Bottom Right) - Simplified ===
    status_width = 200
    status_items = [
        ("STATUS", None),
        ("Debug", state.show_debug),
        ("Depth", state.show_depth_map),
        ("Detect", state.show_detections),
        ("GPT", state.show_gpt_overlay),
        ("Audio", state.audio_enabled),
        ("Smart", state.smart_mode),
        ("Spatial", state.spatial_mode),
    ]
    
    metrics_items = [
        ("FPS", f"{state.metrics.fps:.1f}"),
        ("Infer", f"{state.get_time_until_next_inference():.1f}s"),
        ("Danger", f"{state.current_danger_score:.2f}"),
    ]
    
    status_height = (len(status_items) + len(metrics_items)) * 20 + 30
    status_x = w - status_width - 10
    status_y = h - status_height - 10
    
    _draw_panel(frame, status_x, status_y, status_width, status_height,
                ui['bg_dark'], ui['primary_dark'])
    
    # Draw status items
    y_offset = status_y + 20
    for label, value in status_items:
        if value is None:  # Header
            cv2.putText(frame, label, (status_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts['heading_scale'],
                       ui['accent'], fonts['heading_thickness'])
        else:
            cv2.putText(frame, label, (status_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
                       ui['text_secondary'], fonts['body_thickness'])
            
            # Simple ON/OFF text
            status_text = "ON" if value else "OFF"
            status_color = ui['on'] if value else ui['off']
            text_x = status_x + status_width - 40
            cv2.putText(frame, status_text, (text_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts['small_scale'],
                       status_color, fonts['small_thickness'])
        y_offset += 20
    
    # Divider
    cv2.line(frame, (status_x + 15, y_offset), (status_x + status_width - 15, y_offset),
            ui['primary_dark'], 1)
    y_offset += 10
    
    # Draw metrics
    for label, value in metrics_items:
        cv2.putText(frame, label, (status_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
                   ui['text_secondary'], fonts['body_thickness'])
        
        value_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'], fonts['body_thickness'])[0]
        value_x = status_x + status_width - 15 - value_size[0]
        
        # Simple color coding
        if label == "Danger":
            value_color = ui['danger'] if state.current_danger_score >= 0.7 else ui['success']
        elif label == "FPS":
            value_color = ui['success'] if state.metrics.fps > 20 else ui['warning']
        else:
            value_color = ui['info']
        
        cv2.putText(frame, value, (value_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
                   value_color, fonts['body_thickness'])
        y_offset += 20
    
    return frame


def add_smart_response_overlay(frame: np.ndarray, text: str, danger_score: float) -> np.ndarray:
    """Add smart mode response overlay (optimized - no frame copying).
    
    Args:
        frame: Input frame
        text: Response text to display
        danger_score: Danger score for color coding
        
    Returns:
        Frame with smart response overlay
    """
    # Color based on danger score
    if danger_score >= 0.7:
        color = (0, 0, 255)  # Red
    elif danger_score >= 0.4:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 255, 0)  # Green
    
    # Direct rectangle draw - no frame copying
    cv2.rectangle(
        frame,
        (0, frame.shape[0] - 60),
        (frame.shape[1], frame.shape[0]),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )
    return frame


def add_spatial_overlay(frame: np.ndarray, state: ApplicationState) -> np.ndarray:
    """Add optimized spatial understanding overlay to frame (no frame copying).
    
    Args:
        frame: Input frame
        state: ApplicationState instance
        
    Returns:
        Frame with spatial overlay
    """
    if not state.spatial_mode:
        return frame
    
    ui = config.UI_COLORS
    fonts = config.UI_FONTS
    h, w = frame.shape[:2]
    
    # Direct dark overlay - no frame copying
    cv2.rectangle(frame, (0, 0), (w, h), ui['bg_overlay'], -1)
    
    # Main content panel - simplified
    panel_x = 30
    panel_y = 30
    panel_width = w - 60
    panel_height = h - 60
    
    # Draw panel
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                 ui['bg_dark'], -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                 ui['accent'], 2)
    
    # Title
    title_y = panel_y + 40
    cv2.putText(frame, "SPATIAL UNDERSTANDING",
               (panel_x + 20, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, fonts['title_scale'],
               ui['accent'], fonts['title_thickness'])
    
    # Divider line
    line_y = title_y + 15
    cv2.line(frame, (panel_x + 20, line_y), (panel_x + panel_width - 20, line_y),
            ui['primary'], 1)
    
    # Content area
    content_y = line_y + 30
    content_x = panel_x + 20
    
    if state.spatial_description_text:
        lines = state.spatial_description_text.split('\n')
        y_offset = content_y
        
        for line in lines[:10]:  # Limit to 10 lines max for performance
            if not line.strip():
                y_offset += 12
                continue
            
            # Simple text rendering - limit line length
            line_text = line.strip()[:70]  # Shorter limit
            cv2.putText(frame, line_text,
                       (content_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
                       ui['text_primary'], fonts['body_thickness'])
            y_offset += 25
    else:
        # Loading message
        cv2.putText(frame, "Generating spatial description...",
                   (content_x, content_y),
                   cv2.FONT_HERSHEY_SIMPLEX, fonts['heading_scale'],
                   ui['processing'], fonts['heading_thickness'])
    
    # Footer instruction
    footer_y = panel_y + panel_height - 20
    cv2.putText(frame, "Press 'K' to exit spatial mode",
               (content_x, footer_y),
               cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
               ui['accent'], fonts['body_thickness'])
    
    return frame


def add_deep_spatial_overlay(frame: np.ndarray, state: ApplicationState) -> np.ndarray:
    """Add optimized deep spatial understanding overlay to frame.
    
    Args:
        frame: Input frame (combined camera + depth map)
        state: ApplicationState instance
        
    Returns:
        Frame with deep spatial overlay
    """
    if not state.deep_spatial_mode:
        return frame
    
    ui = config.UI_COLORS
    fonts = config.UI_FONTS
    h, w = frame.shape[:2]
    
    # Draw description panel at bottom (overlay on combined frame)
    panel_height = min(200, h // 3)
    panel_y = h - panel_height
    panel_x = 10
    panel_width = w - 20
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, h - 10),
                 ui['bg_dark'], -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    title_y = panel_y + 30
    cv2.putText(frame, "DEEP SPATIAL UNDERSTANDING",
               (panel_x + 15, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, fonts['title_scale'],
               ui['accent'], fonts['title_thickness'])
    
    # Divider
    line_y = title_y + 15
    cv2.line(frame, (panel_x + 15, line_y), (panel_x + panel_width - 15, line_y),
            ui['primary'], 1)
    
    # Description text
    content_y = line_y + 25
    content_x = panel_x + 15
    
    if state.deep_spatial_description_text:
        lines = state.deep_spatial_description_text.split('\n')
        y_offset = content_y
        
        for line in lines[:8]:  # Limit to 8 lines
            if not line.strip():
                y_offset += 12
                continue
            
            # Simple text rendering
            line_text = line.strip()[:80]  # Limit line length
            cv2.putText(frame, line_text,
                       (content_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
                       ui['text_primary'], fonts['body_thickness'])
            y_offset += 22
    else:
        # Loading message
        cv2.putText(frame, "Generating deep spatial description with depth analysis...",
                   (content_x, content_y),
                   cv2.FONT_HERSHEY_SIMPLEX, fonts['heading_scale'],
                   ui['processing'], fonts['heading_thickness'])
    
    # Footer instruction
    footer_y = h - 15
    cv2.putText(frame, "Press 'J' to exit deep spatial mode",
               (content_x, footer_y),
               cv2.FONT_HERSHEY_SIMPLEX, fonts['body_scale'],
               ui['accent'], fonts['body_thickness'])
    
    return frame


def add_aruco_overlay(frame: np.ndarray, state: ApplicationState) -> np.ndarray:
    """
    Add ArUco detection status overlay.
    
    Args:
        frame: Input frame
        state: ApplicationState instance
    
    Returns:
        Frame with ArUco overlay
    """
    if not state.aruco_enabled:
        return frame
    
    h, w = frame.shape[:2]
    
    # Draw status indicator (top-right corner)
    status_text = "ArUco: ON"
    text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = w - text_size[0] - 10
    text_y = 30
    
    # Background
    cv2.rectangle(
        frame,
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        (0, 100, 0),
        -1
    )
    
    # Text
    cv2.putText(
        frame,
        status_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    
    # Show detection count
    if state.aruco_detections:
        count_text = f"Markers: {len(state.aruco_detections)}"
        count_size, _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        count_x = w - count_size[0] - 10
        count_y = text_y + 25
        
        cv2.putText(
            frame,
            count_text,
            (count_x, count_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    
    return frame

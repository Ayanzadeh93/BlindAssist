#!/usr/bin/env python3
"""Spatial understanding module for AIDGPT application."""
import logging
import time
from typing import Optional
from gpt_interface import GPTInterface
from audio_system import AudioPriority, PriorityAudioQueue
from application_state import ApplicationState


def toggle_spatial_understanding(state: ApplicationState, 
                                audio_queue: PriorityAudioQueue) -> None:
    """Toggle spatial understanding mode and freeze frame for analysis.
    
    Args:
        state: ApplicationState instance
        audio_queue: PriorityAudioQueue instance for audio feedback
    """
    try:
        if not state.spatial_mode:
            # Activate spatial mode and freeze current frame
            state.spatial_mode = True
            state.spatial_frozen = True
            logging.info("Spatial understanding mode activated - frame frozen")
            
            # Test audio to confirm TTS is working
            audio_queue.put_nowait("Spatial understanding mode activated", 
                                  AudioPriority.HIGH, bypass_cooldown=True)
            # Note: Spatial description will be generated in process_frame when frame is captured
        else:
            # Deactivate spatial mode
            state.spatial_mode = False
            state.spatial_frozen = False
            state.spatial_frame = None
            state.spatial_detections = None
            state.spatial_description_text = ""
            logging.info("Spatial understanding mode deactivated")
    except Exception:
        logging.exception("Error toggling spatial understanding")


def generate_spatial_description(state: ApplicationState,
                                gpt_interface: GPTInterface,
                                audio_queue: PriorityAudioQueue) -> None:
    """Generate spatial description using GPT.
    
    Args:
        state: ApplicationState instance
        gpt_interface: GPTInterface instance for querying GPT
        audio_queue: PriorityAudioQueue instance for audio feedback
    """
    try:
        if state.spatial_frame is None or state.spatial_detections is None:
            return
        
        # Create spatial analysis prompt
        spatial_prompt = """You are describing the spatial layout of objects around the user using clock directions and estimated distances. 
Use the following format:

[Object 1] is at your [clock orientation1], about [distance1] away.
[Object 2] is at your [clock orientation2], about [distance2] away.
[Object 3] is at your [clock orientation3], about [distance3] away.
...

Guidelines:
- Clock orientation: relative to the user's forward-facing direction 
  (12 o'clock = straight ahead, 3 o'clock = right, 6 o'clock = directly behind, 9 o'clock = left).
- Distance: express in simple terms like "very close (0–1m)", "near (1–3m)", "medium (3–6m)", or "far (6m+)". 
  If precise values are available, report both (e.g., "about 2.5 meters, near").
- Be concise and consistent.
- List only relevant detected objects.
- If multiple objects share the same orientation, describe them separately.

Example Output:
"The chair is at your 2 o'clock, about 1.5 meters away (near).
The table is at your 12 o'clock, about 3 meters away (medium).
The door is at your 9 o'clock, about 6 meters away (far)."

Analyze the detected objects and provide spatial layout information:"""

        # Get detections info for GPT
        detections_info = []
        for detection in state.spatial_detections:
            detections_info.append(f"- {detection.class_name}: confidence {detection.confidence:.2f}, bbox {detection.bbox}")
        
        if not detections_info:
            detections_info = ["No objects detected in the current frame"]
        
        full_prompt = f"{spatial_prompt}\n\nDetected objects:\n" + "\n".join(detections_info)
        
        # Query GPT for spatial analysis
        gpt_response = gpt_interface.query_gpt(full_prompt)
        
        if gpt_response:
            state.spatial_description_text = gpt_response
            state.spatial_description_display_time = time.monotonic()
            # Queue spatial description as high-priority navigation
            audio_queue.put_nowait(
                gpt_response,
                AudioPriority.HIGH,
                bypass_cooldown=True
            )
            logging.info(f"Spatial description generated and queued for TTS: '{gpt_response[:60]}...'")
        else:
            state.spatial_description_text = "Unable to generate spatial description"
            audio_queue.put_nowait(
                "Unable to generate spatial description",
                AudioPriority.NORMAL,
                bypass_cooldown=True
            )
            
    except Exception:
        logging.exception("Error generating spatial description")
        state.spatial_description_text = "Error generating spatial description"


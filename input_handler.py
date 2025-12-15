#!/usr/bin/env python3
"""Input handling module for AIDGPT application."""
import cv2
import logging
from typing import Callable, Optional
from help import HelpSystem
from audio_system import AudioPriority, PriorityAudioQueue, TTSManager
from application_state import ApplicationState
from spatial_understanding import toggle_spatial_understanding
from deep_spatial_understanding import toggle_deep_spatial_understanding


def create_mouse_callback(state: ApplicationState, help_system: HelpSystem) -> Callable:
    """Create mouse callback function.
    
    Args:
        state: ApplicationState instance
        help_system: HelpSystem instance
        
    Returns:
        Mouse callback function
    """
    def mouse_callback(event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            state.show_debug = not state.show_debug
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.paused = not state.paused
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Handle mouse scroll for help system
            delta = 1 if flags > 0 else -1
            help_system.handle_mouse_scroll(delta)
    
    return mouse_callback


def handle_keyboard(state: ApplicationState,
                   help_system: HelpSystem,
                   audio_queue: PriorityAudioQueue,
                   tts_manager: Optional[TTSManager] = None,
                   is_camera_input: bool = True) -> bool:
    """Handle keyboard input to toggle features and exit.
    
    Args:
        state: ApplicationState instance
        help_system: HelpSystem instance
        audio_queue: PriorityAudioQueue instance
        tts_manager: Optional TTSManager for keeping audio toggle in sync
        
    Returns:
        True to continue, False to quit
    """
    try:
        key = cv2.waitKey(1) & 0xFF
        key_actions = {
            ord('q'): ('quit', None),
            ord('p'): ('pause', 'paused'),
            ord('d'): ('debug overlay', 'show_debug'),
            ord('m'): ('depth map', 'show_depth_map'),
            ord('b'): ('detection boxes', 'show_detections'),
            ord('g'): ('GPT overlay', 'show_gpt_overlay'),
            ord('a'): ('audio', 'audio_enabled'),
            ord('t'): ('test audio', None),
            ord('v'): ('detection alerts', 'detection_alert'),
            ord('s'): ('smart mode', 'smart_mode'),
            ord('k'): ('spatial understanding', 'spatial_mode'),
            ord('j'): ('deep spatial understanding', None),
            ord('o'): ('speak last response', None),
            ord('u'): ('ui overlay', 'show_ui_panel'),
            ord('r'): ('aruco detection', 'aruco_enabled'),  # 'R' for ArUco recognition
            ord('l'): ('help', 'show_help'),  # quick help toggle (top-right help panel)
            ord('h'): ('help', 'show_help')
        }
        # Handle help system navigation first
        if help_system.handle_keyboard(key):
            return True
        
        if key in key_actions:
            action_name, state_attr = key_actions[key]
            if action_name == 'quit':
                state.running = False
                logging.info("Application quit requested")
                return False
            elif action_name == 'spatial understanding':
                toggle_spatial_understanding(state, audio_queue)
            elif action_name == 'deep spatial understanding':
                toggle_deep_spatial_understanding(state, audio_queue, is_camera_input)
            elif action_name == 'test audio':
                if not state.audio_enabled:
                    logging.info("Audio test requested while audio is disabled")
                else:
                    audio_queue.put_nowait(
                        "Audio system test. If you hear this, text to speech is working correctly.", 
                        AudioPriority.HIGH, bypass_cooldown=True)
            elif action_name == 'help':
                help_system.toggle_help()
                state.show_help = help_system.show_help
                if state.show_help:
                    audio_queue.put_nowait(
                        "Help window opened. Press 'H' or 'ESC' to close.", AudioPriority.LOW)
                else:
                    audio_queue.put_nowait("Help window closed.", AudioPriority.LOW)
            elif action_name == 'speak last response':
                # Replay last model response (navigation preferred). Force highest priority and bypass cooldown
                if not state.audio_enabled:
                    logging.info("Manual replay requested while audio is disabled")
                else:
                    text = state.last_navigation_text or state.last_response_text
                    if text:
                        audio_queue.put_nowait(text, AudioPriority.CRITICAL, True)
                        logging.info(f"Manual replay queued: '{text[:50]}...' (danger={state.last_response_danger:.2f})")
                    else:
                        logging.info("No response available to replay")
            elif action_name == 'ui overlay':
                new_state = not state.show_ui_panel
                state.show_ui_panel = new_state
                status = "enabled" if new_state else "hidden"
                audio_queue.put_nowait(f"UI overlay {status}", AudioPriority.LOW, True)
            elif action_name == 'aruco detection':
                current_state = getattr(state, 'aruco_enabled', False)
                new_state = not current_state
                setattr(state, 'aruco_enabled', new_state)
                status = "enabled" if new_state else "disabled"
                audio_queue.put_nowait(
                    f"ArUco marker detection {status}", 
                    AudioPriority.LOW
                )
            else:
                current_state = getattr(state, state_attr, False)
                new_state = not current_state
                setattr(state, state_attr, new_state)
                status = "enabled" if new_state else "disabled"
                feedback_message = f"{action_name.title()} {status}"
                if action_name == 'audio':
                    if tts_manager:
                        tts_manager.audio_enabled = new_state
                    if not new_state and audio_queue:
                        audio_queue.clear()
                        logging.info("Audio disabled; cleared pending audio queue")
                    if new_state:
                        audio_queue.put_nowait("Audio enabled", AudioPriority.HIGH, bypass_cooldown=True)
                else:
                    audio_queue.put_nowait(feedback_message, AudioPriority.LOW)
        return True
    except Exception:
        logging.exception("Error in keyboard handling")
        return True


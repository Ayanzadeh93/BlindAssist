#!/usr/bin/env python3
"""Audio system module for AIDGPT application."""
import queue
import time
import logging
from dataclasses import dataclass
from typing import Optional

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    pyttsx3 = None


class AudioPriority:
    """Audio message priorities for blind navigation."""
    CRITICAL = 0  # Immediate danger (stop, obstacle)
    HIGH = 1      # Important (turn, caution)
    NORMAL = 2    # General info (path clear, status)
    LOW = 3       # Non-urgent (system messages)


@dataclass
class AudioMessage:
    """Audio message with priority and metadata."""
    text: str
    priority: int
    bypass_cooldown: bool = False
    timestamp: float = 0.0

    def __post_init__(self):
        # Use monotonic to avoid issues if system clock changes
        self.timestamp = time.monotonic()


class PriorityAudioQueue:
    """Priority queue for audio messages."""
    
    def __init__(self, maxsize: int = 10):
        self.queue = queue.PriorityQueue(maxsize=maxsize)
        self.message_count = 0
    
    def put(self, text: str, priority: int = AudioPriority.NORMAL, 
            bypass_cooldown: bool = False, block: bool = False, timeout: float = None) -> bool:
        """Add message to queue with priority."""
        try:
            msg = AudioMessage(text, priority, bypass_cooldown)
            # Lower priority number = higher priority
            self.queue.put((priority, self.message_count, msg), block=block, timeout=timeout)
            self.message_count += 1
            return True
        except queue.Full:
            return False
    
    def put_nowait(self, text: str, priority: int = AudioPriority.NORMAL, 
                   bypass_cooldown: bool = False) -> bool:
        """Add message without blocking."""
        return self.put(text, priority, bypass_cooldown, block=False)
    
    def get(self, block: bool = True, timeout: float = None) -> Optional[AudioMessage]:
        """Get highest priority message."""
        try:
            _, _, msg = self.queue.get(block=block, timeout=timeout)
            return msg
        except queue.Empty:
            return None
    
    def clear_low_priority(self):
        """Clear all LOW priority messages (useful when urgent message arrives)."""
        temp_queue = []
        while not self.queue.empty():
            try:
                priority, count, msg = self.queue.get_nowait()
                if priority < AudioPriority.LOW:
                    temp_queue.append((priority, count, msg))
            except queue.Empty:
                break
        
        for item in temp_queue:
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                break

    def clear(self) -> None:
        """Remove all queued messages."""
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            # Best-effort clear; ignore underflow races
            pass


class TTSManager:
    """Text-to-speech manager for audio feedback."""
    
    def __init__(self, audio_enabled: bool = True, test_on_init: bool = False):
        """Initialize TTS manager.
        
        Args:
            audio_enabled: Whether audio is enabled
            test_on_init: Run a blocking self-test; defaults to False to avoid hangs
        """
        self.audio_enabled = audio_enabled
        self._test_on_init = test_on_init
        # Start sufficiently in the past so first speech is not throttled
        self.last_audio_time = time.monotonic() - 10.0
        self.tts_engine = None
        
        if not PYTTSX3_AVAILABLE:
            logging.warning("pyttsx3 not available. Audio feedback will be disabled.")
            self.audio_enabled = False
        else:
            self._initialize_tts()
    
    def _initialize_tts(self) -> None:
        """Initialize TTS engine."""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS properties for blind navigation
            self.tts_engine.setProperty('rate', 175)  # Speech speed (125-200, 175 is good)
            self.tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
            
            # Get and set voice
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Use first voice (usually default/male)
                # Change to voices[1] for female voice if available
                self.tts_engine.setProperty('voice', voices[0].id)
                logging.info(f"Using TTS voice: {voices[0].name}")
                
                # Log all available voices for user reference
                logging.info(f"Available voices: {len(voices)}")
                for idx, voice in enumerate(voices):
                    logging.info(f"  Voice {idx}: {voice.name} ({voice.id})")
            
            logging.info("TTS engine initialized successfully")
            
            # Optional startup self-test; disabled by default to avoid blocking in headless/driver-less envs
            if self._test_on_init:
                logging.info("Testing TTS engine...")
                try:
                    self.tts_engine.say("Audio system ready")
                    self.tts_engine.runAndWait()
                    logging.info("TTS test completed successfully")
                except Exception as test_error:
                    logging.warning(f"TTS test failed: {test_error}")
        
        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
            logging.error("Audio feedback will be disabled")
            self.tts_engine = None
            self.audio_enabled = False
    
    def speak(self, text: str, bypass_cooldown: bool = False) -> None:
        """Convert text to speech using pyttsx3 (offline, fast, non-blocking).
        
        Args:
            text: Text to speak
            bypass_cooldown: If True, ignore cooldown timer
        """
        if not self.audio_enabled:
            return
        
        if self.tts_engine is None:
            logging.warning("TTS engine not available - cannot speak")
            return
        
        current_time = time.monotonic()
        if not bypass_cooldown and current_time - self.last_audio_time < 3:
            return
        
        try:
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            self.last_audio_time = current_time
        except Exception as e:
            logging.error(f"TTS error: {e}")
    
    def queue_audio(self, audio_queue: PriorityAudioQueue, text: str, 
                    priority: int = AudioPriority.NORMAL, 
                    bypass_cooldown: bool = False) -> None:
        """Queue audio message with priority for asynchronous playback.
        
        Args:
            audio_queue: PriorityAudioQueue instance
            text: The text to speak
            priority: AudioPriority level (CRITICAL=0, HIGH=1, NORMAL=2, LOW=3)
            bypass_cooldown: If True, ignore cooldown timer
        """
        if not self.audio_enabled:
            return
        
        # Clear low priority messages if this is urgent
        if priority <= AudioPriority.HIGH:
            audio_queue.clear_low_priority()
        
        success = audio_queue.put_nowait(text, priority, bypass_cooldown)
        if not success:
            logging.warning(f"Audio queue full, dropping: {text[:50]}...")
    
    def set_tts_voice(self, voice_index: int = 0) -> None:
        """Change TTS voice (0=default/male, 1=female usually).
        
        Args:
            voice_index: Index of voice to use (0-based)
        """
        if not self.tts_engine:
            return
        
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices and 0 <= voice_index < len(voices):
                self.tts_engine.setProperty('voice', voices[voice_index].id)
                logging.info(f"Changed TTS voice to: {voices[voice_index].name}")
                # Announce the change
                self.speak(f"Voice changed to {voices[voice_index].name.split()[0]}", bypass_cooldown=True)
            else:
                logging.warning(f"Voice index {voice_index} not available. Available: {len(voices) if voices else 0}")
        except Exception as e:
            logging.error(f"Error changing voice: {e}")
    
    def set_tts_speed(self, speed: int = 175) -> None:
        """Change TTS speed.
        
        Args:
            speed: Words per minute (125=slow, 175=normal, 200=fast)
        """
        if not self.tts_engine:
            return
        
        try:
            # Clamp speed to reasonable range
            speed = max(100, min(300, speed))
            self.tts_engine.setProperty('rate', speed)
            logging.info(f"Changed TTS speed to: {speed} words per minute")
            # Announce the change
            self.speak(f"Speech speed set to {speed}", bypass_cooldown=True)
        except Exception as e:
            logging.error(f"Error changing speed: {e}")
    
    def set_tts_volume(self, volume: float = 1.0) -> None:
        """Change TTS volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if not self.tts_engine:
            return
        
        try:
            # Clamp volume to valid range
            volume = max(0.0, min(1.0, volume))
            self.tts_engine.setProperty('volume', volume)
            logging.info(f"Changed TTS volume to: {volume}")
        except Exception as e:
            logging.error(f"Error changing volume: {e}")
    
    def stop(self) -> None:
        """Stop TTS engine."""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                logging.error(f"Error stopping TTS: {e}")


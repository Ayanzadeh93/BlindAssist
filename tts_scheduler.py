#!/usr/bin/env python3
"""TTS Scheduler for AIDGPT."""
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum

from audio_system import AudioPriority, PriorityAudioQueue, TTSManager, AudioMessage


class TTSState(IntEnum):
    IDLE = 0
    SPEAKING = 1
    QUEUED = 2
    COOLDOWN = 3


@dataclass
class TTSMessage:
    text: str
    priority: int = AudioPriority.NORMAL
    timestamp: float = field(default_factory=time.monotonic)
    max_age: float = 4.5
    is_safety_critical: bool = False

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) >= self.max_age

    @property
    def age(self) -> float:
        return time.monotonic() - self.timestamp


class TTSScheduler:
    """Manages TTS audio delivery synchronized with inference cycles."""

    def __init__(self, tts_manager: TTSManager,
                 audio_queue: PriorityAudioQueue,
                 inference_interval: float = 5.0,
                 min_message_gap: float = 0.5):
        self.tts_manager = tts_manager
        self.audio_queue = audio_queue
        self.inference_interval = inference_interval
        self.min_message_gap = min_message_gap

        self._state = TTSState.IDLE
        self._last_speak_time: float = 0.0
        self._last_message: Optional[str] = None
        self._cycle_message_spoken = False

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        logging.info(f"TTSScheduler initialized with {inference_interval}s cycle")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logging.info("TTSScheduler worker started")

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        logging.info("TTSScheduler worker stopped")

    def _worker_loop(self) -> None:
        logging.info("TTS scheduler worker loop started")
        while self._running and not self._stop_event.is_set():
            try:
                audio_msg = self.audio_queue.get(timeout=0.5)
                if audio_msg:
                    logging.debug(f"TTS scheduler processing message: '{audio_msg.text[:50]}...'")
                    self._speak_message(audio_msg)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in TTS scheduler worker loop: {e}")
                continue

    def _speak_message(self, audio_msg: AudioMessage) -> None:
        if not self.tts_manager:
            logging.warning("TTS manager not available")
            return
        if not self.tts_manager.audio_enabled:
            logging.debug("TTS audio is disabled")
            return

        current_time = time.monotonic()
        if not audio_msg.bypass_cooldown:
            if current_time - self._last_speak_time < self.min_message_gap:
                logging.debug(f"Skipping message due to cooldown (gap: {current_time - self._last_speak_time:.2f}s)")
                return

        self._state = TTSState.SPEAKING
        try:
            logging.info(f"Speaking: '{audio_msg.text}'")
            self.tts_manager.speak(audio_msg.text, audio_msg.bypass_cooldown)
            self._last_speak_time = current_time
            self._last_message = audio_msg.text
            self._cycle_message_spoken = True
            logging.debug("TTS speak completed successfully")
        except Exception as e:
            logging.error(f"TTS speak error: {e}")
        finally:
            self._state = TTSState.IDLE

    def queue_message(self, text: str, priority: int = AudioPriority.NORMAL,
                     is_safety_critical: bool = False, is_navigation: bool = False) -> bool:
        """Queue a message for TTS delivery.
        
        Args:
            text: Message text to speak
            priority: Audio priority level
            is_safety_critical: If True, bypass cooldown
            is_navigation: If True, preserve full text (no truncation) and use navigation cooldown
            
        Returns:
            True if message was queued successfully
        """
        if not text or not text.strip():
            return False
            
        # Import config dynamically to avoid circular imports
        try:
            import config
            max_words = config.Config.TTS_MAX_WORDS if not is_navigation else None
        except Exception:
            max_words = 15 if not is_navigation else None
            
        text = self._shorten_message(text, max_words or 999, is_navigation=is_navigation)
        bypass_cooldown = is_safety_critical or priority <= AudioPriority.HIGH or is_navigation
        return self.audio_queue.put_nowait(text, priority, bypass_cooldown)

    def queue_inference_result(self, result) -> bool:
        if result is None:
            return False
        if result.danger_score >= 0.8:
            priority = AudioPriority.CRITICAL
        elif result.danger_score >= 0.5:
            priority = AudioPriority.HIGH
        elif result.danger_score >= 0.3:
            priority = AudioPriority.NORMAL
        else:
            priority = AudioPriority.LOW
        text = result.navigation if result.navigation else result.response_text
        return self.queue_message(text=text, priority=priority,
                                  is_safety_critical=result.is_safety_critical)

    def _shorten_message(self, text: str, max_words: int = 15, is_navigation: bool = False) -> str:
        """Shorten message to max_words, but never truncate navigation.
        
        Args:
            text: Message text to shorten
            max_words: Maximum number of words
            is_navigation: If True, never truncate (navigation messages are always preserved)
            
        Returns:
            Shortened message text
        """
        if not text:
            return ""
        
        # Never truncate navigation messages
        if is_navigation:
            return text
            
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def on_cycle_start(self) -> None:
        self._cycle_message_spoken = False

    def on_cycle_complete(self, results) -> None:
        if not results:
            return
        best_result = None
        highest_priority = AudioPriority.LOW
        for result in results.values():
            if result.is_safety_critical:
                best_result = result
                break
            if result.danger_score >= 0.5:
                priority = AudioPriority.HIGH
            elif result.danger_score >= 0.3:
                priority = AudioPriority.NORMAL
            else:
                priority = AudioPriority.LOW
            if priority < highest_priority:
                highest_priority = priority
                best_result = result
        if best_result:
            self.queue_inference_result(best_result)

    @property
    def state(self) -> TTSState:
        return self._state

    @property
    def time_since_last_speak(self) -> float:
        return time.monotonic() - self._last_speak_time


_tts_scheduler: Optional[TTSScheduler] = None


def get_tts_scheduler(tts_manager: Optional[TTSManager] = None,
                     audio_queue: Optional[PriorityAudioQueue] = None,
                     inference_interval: float = 5.0) -> Optional[TTSScheduler]:
    global _tts_scheduler
    if _tts_scheduler is None and tts_manager and audio_queue:
        _tts_scheduler = TTSScheduler(tts_manager, audio_queue, inference_interval)
    return _tts_scheduler


def reset_tts_scheduler() -> None:
    global _tts_scheduler
    if _tts_scheduler:
        _tts_scheduler.stop()
    _tts_scheduler = None



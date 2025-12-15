#!/usr/bin/env python3
"""Central time-based inference scheduler."""
import time
import logging
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from enum import Enum
from audio_system import AudioPriority, PriorityAudioQueue


class InferenceState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class InferenceResult:
    model_name: str
    response_text: str
    danger_score: float = 0.0
    navigation: str = ""
    raw_data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.monotonic)
    processing_time: float = 0.0
    is_safety_critical: bool = False

    def __post_init__(self):
        if self.danger_score >= 0.7:
            self.is_safety_critical = True


class ModelAdapter:
    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def is_available(self) -> bool:
        raise NotImplementedError

    async def run_inference(self, frame: np.ndarray, detections: List[Any],
                            frame_info: Dict[str, Any]) -> Optional[InferenceResult]:
        raise NotImplementedError


class GPTModelAdapter(ModelAdapter):
    def __init__(self, gpt_interface, model_name: str = "gpt-4o"):
        self._gpt_interface = gpt_interface
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        try:
            from model_manager import is_model_available
            return is_model_available(self._model_name)
        except Exception:
            return False

    async def run_inference(self, frame: np.ndarray, detections: List[Any],
                            frame_info: Dict[str, Any]) -> Optional[InferenceResult]:
        start_time = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._gpt_interface.process_frame(frame_info, model=self._model_name)
            )
            if response:
                processing_time = time.monotonic() - start_time
                return InferenceResult(
                    model_name=self._model_name,
                    response_text=response.reason,
                    danger_score=response.danger_score,
                    navigation=response.navigation or "",
                    raw_data={"full_response": response},
                    processing_time=processing_time
                )
        except Exception as e:
            logging.error(f"GPT inference error for {self._model_name}: {e}")
        return None


class DepthModelAdapter(ModelAdapter):
    def __init__(self, depth_estimator):
        self._depth_estimator = depth_estimator

    @property
    def model_name(self) -> str:
        return "depth-anything-v2"

    @property
    def is_available(self) -> bool:
        return self._depth_estimator.is_available()

    async def run_inference(self, frame: np.ndarray, detections: List[Any],
                            frame_info: Dict[str, Any]) -> Optional[InferenceResult]:
        start_time = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            depth_map = await loop.run_in_executor(None, lambda: self._depth_estimator.estimate_depth(frame))
            if depth_map is not None:
                processing_time = time.monotonic() - start_time
                return InferenceResult(
                    model_name=self.model_name,
                    response_text="Depth map generated",
                    danger_score=0.0,
                    raw_data={"depth_map": depth_map},
                    processing_time=processing_time
                )
        except Exception as e:
            logging.error(f"Depth estimation error: {e}")
        return None


class InferenceScheduler:
    def __init__(self, inference_interval: float = 5.0,
                 audio_queue: Optional[PriorityAudioQueue] = None):
        self.inference_interval = inference_interval
        self.audio_queue = audio_queue

        self._last_inference_time: float = 0.0
        self._cycle_count: int = 0
        self._audio_enabled_getter: Optional[Callable[[], bool]] = None

        self._models: Dict[str, ModelAdapter] = {}
        self._enabled_models: set = set()

        self._latest_results: Dict[str, InferenceResult] = {}
        self._result_lock = threading.Lock()

        self._current_frame = None
        self._current_detections: List[Any] = []
        self._current_frame_info: Dict[str, Any] = {}
        self._frame_lock = threading.Lock()

        self._result_callbacks: List[Callable[[Dict[str, InferenceResult]], None]] = []

        logging.info(f"InferenceScheduler initialized with {inference_interval}s interval")

    def register_model(self, adapter: ModelAdapter, enabled: bool = True) -> None:
        self._models[adapter.model_name] = adapter
        if enabled:
            self._enabled_models.add(adapter.model_name)
        logging.info(f"Registered model: {adapter.model_name} (enabled={enabled})")

    def add_result_callback(self, callback: Callable[[Dict[str, InferenceResult]], None]) -> None:
        self._result_callbacks.append(callback)

    def set_audio_enabled_getter(self, getter: Callable[[], bool]) -> None:
        """Register a callable that returns whether audio is currently enabled."""
        self._audio_enabled_getter = getter

    def update_frame(self, frame: np.ndarray, detections: List[Any],
                     frame_info: Dict[str, Any]) -> None:
        # Use non-blocking lock to avoid blocking main thread
        # If lock is held (inference running), skip update - previous frame is fine
        acquired = False
        try:
            acquired = self._frame_lock.acquire(blocking=False)
            if acquired:
                # Always copy frame (needed for async processing), but optimize list/dict copies
                self._current_frame = frame.copy() if frame is not None else None
                # Lists and dicts are small - direct assignment is fine (they won't be modified)
                self._current_detections = detections if detections else []
                self._current_frame_info = frame_info if frame_info else {}
        finally:
            if acquired:
                self._frame_lock.release()

    def should_run_inference(self) -> bool:
        current_time = time.monotonic()
        return (current_time - self._last_inference_time) >= self.inference_interval

    async def _run_all_models(self) -> Dict[str, InferenceResult]:
        results = {}
        with self._frame_lock:
            frame = self._current_frame
            detections = self._current_detections
            frame_info = self._current_frame_info
        if frame is None:
            return results

        tasks = []
        model_names = []
        for model_name in self._enabled_models:
            adapter = self._models.get(model_name)
            if adapter and adapter.is_available:
                tasks.append(adapter.run_inference(frame, detections, frame_info))
                model_names.append(model_name)

        if not tasks:
            return results

        try:
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=4.5
            )
            for model_name, result in zip(model_names, completed):
                if isinstance(result, Exception):
                    logging.error(f"Model {model_name} failed: {result}")
                elif result is not None:
                    results[model_name] = result
        except asyncio.TimeoutError:
            logging.warning("Inference cycle timed out")
        return results

    def run_inference_cycle(self) -> Dict[str, InferenceResult]:
        self._last_inference_time = time.monotonic()
        self._cycle_count += 1

        try:
            results = asyncio.run(self._run_all_models())
        except Exception as e:
            logging.error(f"Inference cycle error: {e}")
            results = {}

        with self._result_lock:
            self._latest_results = results

        for callback in self._result_callbacks:
            try:
                callback(results)
            except Exception as e:
                logging.error(f"Result callback error: {e}")

        self._queue_audio_for_results(results)
        return results

    def _queue_audio_for_results(self, results: Dict[str, InferenceResult]) -> None:
        """Queue audio messages for inference results with enhanced navigation handling.
        
        Improvements:
        - Speak navigation separately from danger assessment
        - Never truncate navigation instructions
        - Use proper priority and cooldown for navigation
        """
        if not self.audio_queue or not results:
            return
        if self._audio_enabled_getter and not self._audio_enabled_getter():
            return
            
        # Import config for TTS settings
        try:
            import config
            speak_nav_separately = config.Config.TTS_SPEAK_NAVIGATION_SEPARATELY
            critical_threshold = config.Config.TTS_CRITICAL_DANGER_THRESHOLD
            nav_prefix = config.Config.TTS_NAVIGATION_PREFIX
        except Exception:
            speak_nav_separately = True
            critical_threshold = 0.5
            nav_prefix = "Navigation:"
            
        best_result: Optional[InferenceResult] = None
        highest_priority = AudioPriority.LOW
        
        # Find the best result
        for result in results.values():
            if result.is_safety_critical or result.danger_score >= 0.8:
                priority = AudioPriority.CRITICAL
            elif result.danger_score >= 0.5:
                priority = AudioPriority.HIGH
            elif result.danger_score >= 0.3:
                priority = AudioPriority.NORMAL
            else:
                priority = AudioPriority.LOW
                
            if best_result is None or priority < highest_priority:
                highest_priority = priority
                best_result = result
        
        if not best_result:
            return
            
        # Handle navigation-specific logic
        has_navigation = bool(best_result.navigation and best_result.navigation.strip())
        has_response = bool(best_result.response_text and best_result.response_text.strip())
        
        if speak_nav_separately and has_navigation:
            # Speak danger assessment first (if significant danger)
            if has_response and best_result.danger_score >= 0.3:
                danger_priority = AudioPriority.CRITICAL if best_result.danger_score >= critical_threshold else AudioPriority.HIGH
                danger_bypass = best_result.danger_score >= critical_threshold or best_result.is_safety_critical
                self.audio_queue.put_nowait(
                    best_result.response_text.strip(),
                    danger_priority,
                    danger_bypass
                )
                logging.info(f"Queued danger assessment: '{best_result.response_text[:50]}...'")
            
            # Then speak navigation separately with prefix
            nav_text = f"{nav_prefix} {best_result.navigation.strip()}"
            nav_priority = AudioPriority.CRITICAL if best_result.danger_score >= critical_threshold else AudioPriority.HIGH
            # Navigation always bypasses cooldown
            self.audio_queue.put_nowait(
                nav_text,
                nav_priority,
                bypass_cooldown=True
            )
            logging.info(f"Queued navigation: '{nav_text[:50]}...'")
            
        elif has_navigation:
            # Fallback: just speak navigation (preferred over response_text)
            nav_priority = AudioPriority.CRITICAL if best_result.danger_score >= critical_threshold else AudioPriority.HIGH
            self.audio_queue.put_nowait(
                best_result.navigation.strip(),
                nav_priority,
                bypass_cooldown=True
            )
            logging.info(f"Queued navigation only: '{best_result.navigation[:50]}...'")
            
        elif has_response:
            # No navigation, just speak response
            response_priority = highest_priority
            response_bypass = highest_priority <= AudioPriority.HIGH or best_result.is_safety_critical
            self.audio_queue.put_nowait(
                best_result.response_text.strip(),
                response_priority,
                response_bypass
            )
            logging.info(f"Queued response: '{best_result.response_text[:50]}...'")


    def get_primary_response(self) -> Optional[InferenceResult]:
        with self._result_lock:
            if not self._latest_results:
                return None
            critical = None
            highest_danger = 0.0
            for result in self._latest_results.values():
                if result.is_safety_critical and result.danger_score > highest_danger:
                    critical = result
                    highest_danger = result.danger_score
            if critical:
                return critical
            for model_name in ["gpt-4o", "gpt-5", "claude-3.5", "llama-3.2-vision"]:
                if model_name in self._latest_results:
                    return self._latest_results[model_name]
            if self._latest_results:
                return next(iter(self._latest_results.values()))
            return None

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cycle_count": self._cycle_count,
            "inference_interval": self.inference_interval,
            "registered_models": list(self._models.keys()),
            "enabled_models": list(self._enabled_models),
            "results_available": len(self._latest_results),
        }


_scheduler: Optional[InferenceScheduler] = None


def get_inference_scheduler(inference_interval: float = 5.0,
                           audio_queue: Optional[PriorityAudioQueue] = None) -> InferenceScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = InferenceScheduler(inference_interval, audio_queue)
    return _scheduler


def reset_scheduler() -> None:
    global _scheduler
    _scheduler = None




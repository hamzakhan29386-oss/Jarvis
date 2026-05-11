"""
wake/detector.py — OpenWakeWord Detection Engine
==================================================
Wraps openwakeword with:
  - Rolling confidence window (reduces single-frame false positives)
  - Configurable threshold + cooldown
  - Per-model score logging
  - Hot-reload of custom models

Requirements:
    pip install openwakeword
"""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from .config import (
    WAKE_MODELS, WAKE_THRESHOLD,
    WAKE_COOLDOWN_SECONDS, WAKE_ROLLING_WINDOW,
    LOG_DETECTIONS,
)

log = logging.getLogger("jarvis.wake.detector")


class WakeWordEngine:
    """
    OpenWakeWord-based detector with rolling confirmation window
    and cooldown logic to prevent re-triggering within N seconds.
    """

    def __init__(
        self,
        models: list = None,
        threshold: float = WAKE_THRESHOLD,
        cooldown_s: float = WAKE_COOLDOWN_SECONDS,
        rolling_window: int = WAKE_ROLLING_WINDOW,
    ):
        self._models_cfg = models or WAKE_MODELS
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._rolling_window = rolling_window

        self._model = None
        self._available = False
        self._last_detection: float = 0.0   # epoch time of last confirmed detection

        # Per-keyword rolling score windows: {keyword: deque([score, ...])}
        self._score_windows: dict[str, deque] = {}

        self._init()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init(self):
        try:
            from openwakeword.model import Model
            self._model = Model(
                wakeword_models=self._models_cfg,
                inference_framework="onnx",
            )
            # Seed score windows for every model key
            for key in self._model.prediction_buffer.keys():
                self._score_windows[key] = deque(maxlen=self._rolling_window)
            self._available = True
            log.info(
                f"[Detector] OpenWakeWord ready | "
                f"models={self._models_cfg} | threshold={self._threshold}"
            )
        except ImportError:
            log.error(
                "[Detector] openwakeword not installed. "
                "Run: pip install openwakeword"
            )
        except Exception as e:
            log.error(f"[Detector] Init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    # ── Detection ────────────────────────────────────────────────────────────

    def process_chunk(self, chunk_np: np.ndarray) -> tuple[bool, str, float]:
        """
        Process one audio chunk and return whether the wake word was detected.

        Args:
            chunk_np: int16 numpy array, exactly CHUNK_SAMPLES long.

        Returns:
            (detected: bool, keyword: str, score: float)
            If detected=False, keyword="" and score is the highest score seen.
        """
        if not self._available:
            return False, "", 0.0

        # Cooldown check
        now = time.monotonic()
        if now - self._last_detection < self._cooldown_s:
            return False, "", 0.0

        try:
            predictions = self._model.predict(chunk_np)
            print(predictions)
        except Exception as e:
            log.debug(f"[Detector] predict error: {e}")
            return False, "", 0.0

        best_key = ""
        best_score = 0.0
        detected = False

        for key, score in predictions.items():
            # Maintain rolling window
            if key not in self._score_windows:
                self._score_windows[key] = deque(maxlen=self._rolling_window)
            self._score_windows[key].append(float(score))

            if LOG_DETECTIONS and score > 0.1:
                log.debug(f"[Detector] {key}: {score:.3f}")

            # Only confirm when the rolling average exceeds threshold
            window = self._score_windows[key]
            if len(window) == self._rolling_window:
                avg_score = sum(window) / len(window)
                if avg_score >= self._threshold and avg_score > best_score:
                    best_score = avg_score
                    best_key = key
                    detected = True

        if detected:
            self._last_detection = now
            log.info(
                f"[Detector] ✓ WAKE WORD DETECTED | "
                f"keyword={best_key} | score={best_score:.3f}"
            )
            self._reset_windows()

        return detected, best_key, best_score

    def _reset_windows(self):
        """Clear rolling windows after a detection to avoid immediate re-trigger."""
        for key in self._score_windows:
            self._score_windows[key].clear()

    # ── Config ───────────────────────────────────────────────────────────────

    def set_threshold(self, threshold: float):
        self._threshold = max(0.0, min(1.0, threshold))
        log.info(f"[Detector] Threshold updated → {self._threshold:.2f}")

    def set_cooldown(self, seconds: float):
        self._cooldown_s = max(0.0, seconds)

    def load_custom_model(self, model_path: str):
        """Hot-reload a custom .onnx model without restarting."""
        self._models_cfg = [model_path]
        self._model = None
        self._available = False
        self._score_windows.clear()
        self._init()

    def get_status(self) -> dict:
        return {
            "available": self._available,
            "models": self._models_cfg,
            "threshold": self._threshold,
            "cooldown_s": self._cooldown_s,
            "rolling_window": self._rolling_window,
            "last_detection_ago_s": (
                round(time.monotonic() - self._last_detection, 1)
                if self._last_detection else None
            ),
        }

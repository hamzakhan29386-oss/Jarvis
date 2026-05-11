"""
wake/service.py — Production Wake Word Service for JARVIS Flask Backend
========================================================================
Drop-in replacement for the WakeWordService in voice.py.
Integrates the production WakeEngine with the existing SSE event system.

Improvements over original:
    - Uses WakeEngine (proper pipeline: VAD → NS → OWW → SV)
    - Thread-safe subscriber queue
    - Automatic reconnect on mic failure
    - Faster Whisper STT integration
    - Speaker verification gate
    - Proper cooldown management
    - Structured event payloads
    - Health status reporting

Integration with server.py:
    Replace:  from voice import get_wake_service
    With:     from wake.service import get_wake_service
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.wake.service")

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "hey_jarvis_custom.onnx"


# ─── STT (Faster-Whisper) ─────────────────────────────────────────────────────

class STTEngine:
    """Faster-Whisper STT. Transcribes command audio after wake detection."""

    def __init__(self, model_size: str = "tiny.en"):
        self._model = None
        self._model_size = model_size
        self._lock = threading.Lock()

    def _init(self):
        if self._model:
            return
        try:
            from faster_whisper import WhisperModel
            log.info("[STT] Loading Whisper %s...", self._model_size)
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type="int8"
            )
            log.info("[STT] Ready")
        except ImportError:
            log.warning("[STT] faster-whisper not installed — pip install faster-whisper")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16_000) -> str:
        """Transcribe raw int16 PCM bytes to text."""
        with self._lock:
            self._init()
            if not self._model:
                return ""
            try:
                import wave, tempfile, numpy as np
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_bytes)
                segments, _ = self._model.transcribe(
                    tmp.name, beam_size=3, language="en", vad_filter=True
                )
                text = " ".join(s.text for s in segments).strip()
                import os; os.unlink(tmp.name)
                return text
            except Exception as e:
                log.error("[STT] Error: %s", e)
                return ""


# ─── Command capture ──────────────────────────────────────────────────────────

class CommandCapture:
    """
    Captures speech after wake word detection.
    Listens until silence, then returns audio bytes for STT.
    """

    SAMPLE_RATE     = 16_000
    MAX_SECS        = 8.0
    SILENCE_SECS    = 1.2
    MIN_RMS         = 100

    def capture(self) -> Optional[bytes]:
        """
        Record until the user stops speaking.
        Returns raw int16 PCM bytes, or None on failure.
        """
        try:
            import sounddevice as sd
            import numpy as np

            frames       = []
            silence_start = None
            start_time    = time.time()

            def cb(indata, f, t, status):
                frames.append(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

            with sd.InputStream(
                samplerate = self.SAMPLE_RATE,
                channels   = 1,
                dtype      = "int16",
                blocksize  = 1024,
                callback   = cb,
            ):
                while True:
                    time.sleep(0.05)
                    elapsed = time.time() - start_time
                    if elapsed > self.MAX_SECS:
                        break
                    if not frames:
                        continue
                    chunk = frames[-1].astype(np.float32)
                    rms   = float(np.sqrt(np.mean(chunk ** 2)))
                    if rms > self.MIN_RMS:
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.SILENCE_SECS:
                            break

            if not frames:
                return None

            audio = np.concatenate(frames).flatten()
            return audio.tobytes()

        except Exception as e:
            log.error("[Capture] Error: %s", e)
            return None


# ─── Production Wake Service ──────────────────────────────────────────────────

class ProductionWakeService:
    """
    Production-grade wake word service.
    Manages WakeEngine + STT + SSE event distribution.

    Events published to subscribers:
        {"event": "detected",    "confidence": float}
        {"event": "listening"}
        {"event": "transcript",  "text": str, "latency_ms": float}
        {"event": "error",       "message": str}
        {"event": "heartbeat"}
        {"event": "disabled"}
        {"event": "unavailable", "message": str}
    """

    def __init__(self):
        self._enabled      = False
        self._engine       = None
        self._stt          = STTEngine()
        self._capture      = CommandCapture()
        self._stop         = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribers: list[queue.Queue] = []
        self._sub_lock     = threading.Lock()
        self._current_event: Optional[str] = None

    # ── Pub/Sub ───────────────────────────────────────────────────────────

    def subscribe(self):
        """Generator yielding event dicts. Each caller gets its own queue."""
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield {"event": "heartbeat"}
        finally:
            with self._sub_lock:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    def _publish(self, event: dict):
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def enable(self):
        if self._enabled:
            return
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WakeService"
        )
        self._thread.start()
        log.info("[WakeService] Enabled")

    def disable(self):
        self._enabled = False
        self._stop.set()
        if self._engine:
            self._engine.stop()
            self._engine = None
        log.info("[WakeService] Disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def get_status(self) -> dict:
        return {
            "enabled":     self._enabled,
            "engine_ready": self._engine is not None and self._engine.ready,
            "model":       str(MODEL_PATH) if MODEL_PATH.exists() else "built-in hey_jarvis_v0.1",
            "subscribers": len(self._subscribers),
        }

    # ── Main loop ─────────────────────────────────────────────────────────

    def _run(self):
        """Start engine and process wake events until disabled."""
        try:
            from wake.engine import WakeEngine, EngineConfig
        except ImportError:
            try:
                from engine import WakeEngine, EngineConfig  # relative import fallback
            except ImportError:
                msg = "WakeEngine not found — ensure wake/engine.py is on PYTHONPATH"
                log.error(msg)
                self._publish({"event": "unavailable", "message": msg})
                return

        # Determine model path
        model_path = str(MODEL_PATH) if MODEL_PATH.exists() else None

        config = EngineConfig(
            threshold      = 0.50,   # production threshold (tuned)
            rolling_window = 3,
            cooldown_secs  = 2.5,
            model_path     = model_path,
            debug_scores   = False,
        )

        try:
            self._engine = WakeEngine(config=config)
            if not self._engine.ready:
                raise RuntimeError("Model load failed")
        except Exception as e:
            log.error("[WakeService] Engine init failed: %s", e)
            self._publish({"event": "unavailable", "message": str(e)})
            return

        self._engine.on_wake = self._handle_wake
        self._engine.start()

        log.info("[WakeService] Engine running. Model=%s  Threshold=%.2f",
                 model_path or "hey_jarvis_v0.1", config.threshold)

        # Block until disabled
        self._stop.wait()
        self._engine.stop()
        log.info("[WakeService] Engine stopped")

    def _handle_wake(self, event):
        """Called by WakeEngine on every confirmed detection."""
        from wake.engine import DetectionEvent

        if not self._enabled:
            return

        log.info("[WakeService] Wake detected (conf=%.3f)", event.confidence)
        self._publish({
            "event":      "detected",
            "confidence": round(event.confidence, 3),
            "latency_ms": round(event.latency_ms, 1),
        })

        # Give hardware time to settle before reopening mic
        time.sleep(0.15)

        # Signal frontend to show listening UI
        self._publish({"event": "listening"})

        # Capture command audio
        t_stt_start = time.time()
        audio_bytes = self._capture.capture()

        if not audio_bytes:
            self._publish({"event": "error", "message": "Didn't catch that. Try again."})
            return

        # Transcribe
        text = self._stt.transcribe(audio_bytes)
        latency_ms = (time.time() - t_stt_start) * 1000

        if text and text.strip():
            log.info("[WakeService] Transcript: %s  (%.0fms)", text, latency_ms)
            self._publish({
                "event":      "transcript",
                "text":       text.strip(),
                "latency_ms": round(latency_ms),
            })
        else:
            self._publish({
                "event":   "error",
                "message": "Didn't catch that — please try again.",
            })


# ─── Singleton ────────────────────────────────────────────────────────────────

_service_instance:  Optional[ProductionWakeService] = None
_service_lock       = threading.Lock()


def get_wake_service() -> ProductionWakeService:
    """Return the global ProductionWakeService singleton."""
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ProductionWakeService()
    return _service_instance

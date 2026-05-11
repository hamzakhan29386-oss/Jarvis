"""
wake/service.py — JARVIS Wake Word Service (Production)
=========================================================
Orchestrates the full pipeline:

    Mic → NoisePipeline → WakeWordEngine → SpeakerVerifier
        → PostWakeTranscriber → SSE event queue → Frontend

Architecture:
  - One long-lived background thread runs the microphone loop
  - Watchdog thread monitors for hangs and restarts if needed
  - SSE subscribers receive events from a per-client queue
  - Thread-safe, safe to call enable/disable from any thread

This replaces the WakeWordService in the original voice.py.
Drop in by updating voice.py's get_wake_service() to return this.

Usage (from server.py):
    from wake.service import ProductionWakeService, get_wake_service
    svc = get_wake_service()
    svc.enable()
    for event in svc.subscribe():
        yield f"data: {json.dumps(event)}\n\n"
"""

import logging
import queue
import threading
import time
from typing import Iterator

import numpy as np

from .config import (
    SAMPLE_RATE, CHUNK_SAMPLES, DTYPE,
    SPEAKER_VERIFY_ENABLED,
    WATCHDOG_INTERVAL_S, MAX_RESTART_ATTEMPTS,
    STREAM_TIMEOUT_S,
)
from .noise import NoisePipeline
from .detector import WakeWordEngine
from .speaker import SpeakerVerifier
from .transcriber import PostWakeTranscriber

log = logging.getLogger("jarvis.wake.service")


# ── Event helpers ─────────────────────────────────────────────────────────────

def _evt(name: str, **kw) -> dict:
    return {"event": name, "ts": time.time(), **kw}


class ProductionWakeService:
    """
    Production-quality always-listening wake word service.

    Lifecycle:
        svc.enable()   → starts background thread + watchdog
        svc.disable()  → stops everything, unblocks all subscribers
        svc.subscribe() → generator that yields event dicts
    """

    def __init__(self):
        self._enabled = False
        self._stop = threading.Event()

        # Components (lazy-init on first enable)
        self._noise: NoisePipeline | None = None
        self._detector: WakeWordEngine | None = None
        self._speaker: SpeakerVerifier | None = None
        self._transcriber: PostWakeTranscriber | None = None

        # SSE pub/sub
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()

        # Thread management
        self._listener_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._last_heartbeat: float = 0.0
        self._restart_count: int = 0

        # State
        self._in_command_capture: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def enable(self):
        """Start the background listener and watchdog threads."""
        if self._enabled:
            return
        self._stop.clear()
        self._enabled = True
        self._restart_count = 0
        self._init_components()
        self._start_listener()
        self._start_watchdog()
        log.info("[WakeSvc] Service enabled")

    def disable(self):
        """Stop everything and notify all subscribers."""
        self._enabled = False
        self._stop.set()
        self._publish(_evt("disabled"))
        log.info("[WakeSvc] Service disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def subscribe(self) -> Iterator[dict]:
        """
        Generator — yields event dicts as they arrive.
        Each caller gets its own queue. Auto-cleans up on generator close.
        Heartbeats emitted every STREAM_TIMEOUT_S to keep SSE connections alive.
        """
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        log.debug("[WakeSvc] New SSE subscriber")
        try:
            while True:
                try:
                    yield q.get(timeout=STREAM_TIMEOUT_S)
                except queue.Empty:
                    yield _evt("heartbeat")
        finally:
            with self._sub_lock:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass
            log.debug("[WakeSvc] SSE subscriber disconnected")

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "in_command_capture": self._in_command_capture,
            "restart_count": self._restart_count,
            "detector": self._detector.get_status() if self._detector else None,
            "speaker": self._speaker.get_status() if self._speaker else None,
            "transcriber": self._transcriber.get_status() if self._transcriber else None,
        }

    def set_threshold(self, threshold: float):
        if self._detector:
            self._detector.set_threshold(threshold)

    def enroll_speaker(self):
        """Convenience wrapper — trigger speaker enrollment from server.py."""
        if self._speaker:
            return self._speaker.enroll_from_mic()
        return False

    # ── Component init ────────────────────────────────────────────────────────

    def _init_components(self):
        if self._noise is None:
            self._noise = NoisePipeline()
        if self._detector is None:
            self._detector = WakeWordEngine()
        if self._speaker is None:
            self._speaker = SpeakerVerifier()
        if self._transcriber is None:
            self._transcriber = PostWakeTranscriber()

    # ── Thread management ─────────────────────────────────────────────────────

    def _start_listener(self):
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="JARVISWakeListener",
        )
        self._listener_thread.start()

    def _start_watchdog(self):
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="JARVISWakeWatchdog",
        )
        self._watchdog_thread.start()

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog_loop(self):
        """Monitor the listener thread; restart on hangs or crashes."""
        log.info("[WakeSvc] Watchdog started")
        time.sleep(WATCHDOG_INTERVAL_S)

        while not self._stop.is_set() and self._enabled:
            alive = (
                self._listener_thread is not None
                and self._listener_thread.is_alive()
            )
            stale = (
                self._last_heartbeat > 0
                and time.monotonic() - self._last_heartbeat > WATCHDOG_INTERVAL_S * 2
            )

            if not alive or stale:
                if self._restart_count >= MAX_RESTART_ATTEMPTS:
                    log.error("[WakeSvc] Max restarts exceeded — giving up")
                    self._publish(_evt("error", message="Wake service could not recover"))
                    self.disable()
                    return

                reason = "crashed" if not alive else "hung"
                log.warning(
                    f"[WakeSvc] Listener {reason} — "
                    f"restarting ({self._restart_count + 1}/{MAX_RESTART_ATTEMPTS})"
                )
                self._restart_count += 1
                self._publish(_evt("error", message=f"Listener {reason}, restarting..."))
                self._start_listener()

            time.sleep(WATCHDOG_INTERVAL_S)

    # ── Main listener loop ────────────────────────────────────────────────────

    def _listener_loop(self):
        """
        Continuously open the microphone and process audio through the pipeline.
        Runs until self._stop is set.
        """
        log.info("[WakeSvc] Listener started")

        try:
            import sounddevice as sd
        except ImportError:
            msg = "sounddevice not installed. Run: pip install sounddevice"
            log.error(f"[WakeSvc] {msg}")
            self._publish(_evt("unavailable", message=msg))
            return

        if not (self._detector and self._detector.available):
            msg = "OpenWakeWord not available — install openwakeword"
            log.error(f"[WakeSvc] {msg}")
            self._publish(_evt("unavailable", message=msg))
            return

        log.info("[WakeSvc] Listening for wake word...")

        while not self._stop.is_set():
            try:
                self._run_detection_pass(sd)
            except Exception as e:
                if not self._stop.is_set():
                    log.error(f"[WakeSvc] Detection pass error: {e}")
                    time.sleep(1.0)

        log.info("[WakeSvc] Listener stopped")

    def _run_detection_pass(self, sd):
        """
        One complete detection pass:
        open stream → listen for wake word → close stream → capture command.
        Uses a separate stream for each phase to avoid resource conflicts.
        """
        buf_ready = threading.Event()
        latest_np = [None]

        def audio_cb(indata, frames, time_info, status):
            raw = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
            latest_np[0] = raw
            buf_ready.set()

        # ── Phase 1: Wake word detection ──────────────────────────────────────
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=DTYPE,
            blocksize=CHUNK_SAMPLES,
            callback=audio_cb,
        ):
            while not self._stop.is_set():
                self._last_heartbeat = time.monotonic()

                if not buf_ready.wait(timeout=1.0):
                    continue   # no audio yet — loop
                buf_ready.clear()

                chunk_np = latest_np[0]
                if chunk_np is None:
                    continue

                # Noise suppression
                processed_bytes, has_speech, _ = self._noise.process(chunk_np.tobytes())
                if not has_speech:
                    continue

                processed_np = np.frombuffer(processed_bytes, dtype=np.int16)

                # Wake word detection
                detected, keyword, score = self._detector.process_chunk(processed_np)
                if not detected:
                    continue

                # Speaker verification (optional)
                if SPEAKER_VERIFY_ENABLED and self._speaker and self._speaker.enrolled:
                    ok, sim = self._speaker.verify(processed_np)
                    if not ok:
                        log.info(f"[WakeSvc] Speaker rejected (similarity={sim:.3f})")
                        continue
                    log.info(f"[WakeSvc] Speaker verified (similarity={sim:.3f})")

                log.info(f"[WakeSvc] Wake word confirmed: {keyword} ({score:.3f})")
                self._publish(_evt("detected", keyword=keyword, score=round(score, 3)))
                break   # exit wake detection loop → proceed to STT

        if self._stop.is_set():
            return

        # ── Phase 2: Command capture ──────────────────────────────────────────
        self._in_command_capture = True
        self._publish(_evt("listening"))

        # Brief gap so hardware releases from previous stream
        time.sleep(0.15)

        text = self._transcriber.capture_and_transcribe(
            status_callback=lambda e: self._publish(_evt(e))
        )
        self._in_command_capture = False

        if text and text.strip():
            self._publish(_evt("transcript", text=text.strip()))
        else:
            self._publish(_evt("error", message="Didn't catch that — please try again"))

        # Small cooldown before reopening detection stream
        time.sleep(0.3)

    # ── Pub/sub ───────────────────────────────────────────────────────────────

    def _publish(self, event: dict):
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass   # slow consumer — drop rather than block


# ── Singleton ──────────────────────────────────────────────────────────────────

_instance: ProductionWakeService | None = None
_lock = threading.Lock()


def get_wake_service() -> ProductionWakeService:
    """Get or create the global ProductionWakeService singleton."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ProductionWakeService()
    return _instance

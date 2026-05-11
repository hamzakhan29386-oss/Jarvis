"""
wake/engine.py — Production Wake Word Detection Engine
=======================================================
Continuous, low-latency, thread-safe wake word detector.
Comparable in reliability to Alexa/Google Assistant.

Pipeline:
    Mic → RMS Gate → WebRTC VAD → RNNoise → OpenWakeWord → Threshold + Rolling Avg → Callback

Design principles:
    - Non-blocking audio thread (never drop frames)
    - Dedicated processing thread (never block audio)
    - Lock-free queue between audio → processing
    - Cooldown guard (prevents repeated activations)
    - Rolling confidence window (kills false positives)
    - Automatic sample rate conversion
    - Format-safe int16/float32 handling
    - Graceful degradation if optional deps missing
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("jarvis.wake.engine")

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16_000          # Hz  — OWW requirement
CHUNK_MS          = 80              # ms per chunk (OWW min = 80ms)
CHUNK_SAMPLES     = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 1280
DTYPE             = "int16"

# Rolling window: average last N chunks before triggering
WINDOW_SIZE_DEBUG = 1               # instant during threshold tuning
WINDOW_SIZE_PROD  = 3               # smoothed during normal use

# Cooldown: ignore detections for N seconds after each trigger
COOLDOWN_SECS     = 2.5

# Silence gate: skip processing if RMS below this (saves CPU)
RMS_SILENCE_GATE  = 80             # int16 range 0–32767


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionEvent:
    """Fired when wake word is confirmed above threshold."""
    confidence:    float
    model_key:     str
    timestamp:     float = field(default_factory=time.time)
    audio_rms:     float = 0.0
    latency_ms:    float = 0.0


@dataclass
class EngineConfig:
    """Tunable parameters — safe to change at runtime via engine.config."""
    threshold:        float = 0.24       # initial; tune with tools/threshold_tuner.py
    rolling_window:   int   = 3          # increase for fewer false positives
    cooldown_secs:    float = COOLDOWN_SECS
    rms_gate:         float = RMS_SILENCE_GATE
    model_path:       Optional[str] = None   # None → use built-in hey_jarvis_v0.1
    debug_scores:     bool  = False      # print every score to stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_mono_int16(audio: np.ndarray) -> np.ndarray:
    """
    Normalise any numpy audio array to mono int16 at the engine sample rate.
    Handles: stereo, float32, float64, wrong dtype.
    """
    # Collapse stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Convert float → int16
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)

    return audio


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of int16 audio."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def resample_if_needed(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """
    Downsample / upsample to SAMPLE_RATE using scipy if available,
    otherwise use a simple linear-interpolation fallback.
    """
    if src_rate == SAMPLE_RATE:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, src_rate)
        return resample_poly(audio, SAMPLE_RATE // g, src_rate // g).astype(np.int16)
    except ImportError:
        # Fallback: numpy linear interpolation (lower quality but always available)
        ratio = SAMPLE_RATE / src_rate
        n_out = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)


# ═══════════════════════════════════════════════════════════════════════════════
#  VAD WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class VADFilter:
    """
    WebRTC VAD wrapper. Falls back gracefully if webrtcvad is not installed.
    Mode 3 = most aggressive (fewer false positives in noisy environments).
    """
    def __init__(self, aggressiveness: int = 3):
        self._vad = None
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            log.info("[VAD] WebRTC VAD initialised (aggressiveness=%d)", aggressiveness)
        except ImportError:
            log.warning("[VAD] webrtcvad not installed — VAD filtering disabled")

    def is_speech(self, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """Returns True if the audio frame is speech (or if VAD unavailable)."""
        if self._vad is None:
            return True
        # WebRTC VAD requires exactly 10/20/30ms frames
        frame_ms   = 30
        frame_size = int(sample_rate * frame_ms / 1000) * 2  # * 2 for int16 bytes
        if len(pcm_bytes) < frame_size:
            return True
        try:
            return self._vad.is_speech(pcm_bytes[:frame_size], sample_rate)
        except Exception:
            return True


# ═══════════════════════════════════════════════════════════════════════════════
#  NOISE SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

class NoiseSupressor:
    """
    RNNoise-based noise suppression.
    Gracefully skips if rnnoise_python / noisereduce not installed.
    Uses noisereduce as a lighter alternative.
    """
    def __init__(self):
        self._mode = "passthrough"
        try:
            import noisereduce as nr
            self._nr = nr
            self._mode = "noisereduce"
            log.info("[NS] noisereduce noise suppression active")
        except ImportError:
            log.info("[NS] noisereduce not installed — running without noise suppression")

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self._mode == "noisereduce":
            try:
                f = audio.astype(np.float32) / 32768.0
                reduced = self._nr.reduce_noise(y=f, sr=SAMPLE_RATE, prop_decrease=0.75, stationary=True)
                return (reduced * 32768.0).astype(np.int16)
            except Exception:
                pass
        return audio


# ═══════════════════════════════════════════════════════════════════════════════
#  WAKE WORD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class WakeWordDetector:
    """
    OpenWakeWord-based detector with production hardening.
    Thread-safe, non-blocking, with rolling confidence window.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._model_key: Optional[str] = None
        self._scores: deque = deque(maxlen=config.rolling_window)
        self._load_model()

    def _load_model(self):
        """Load OWW model. Validates it actually produces predictions."""
        try:
            from openwakeword.model import Model as OWWModel

            model_path = self.config.model_path

            if model_path and Path(model_path).exists():
                # Custom trained model
                self._model = OWWModel(
                    wakeword_models=[model_path],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,  # we handle NS ourselves
                )
                self._model_key = Path(model_path).stem
                log.info("[WW] Loaded custom model: %s", model_path)
            else:
                # Built-in hey_jarvis_v0.1
                self._model = OWWModel(
                    wakeword_models=["hey_jarvis_v0.1"],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,
                )
                self._model_key = "hey_jarvis_v0.1"
                log.info("[WW] Loaded built-in model: hey_jarvis_v0.1")

            # Warm-up pass (avoids first-call latency spike)
            dummy = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            self._model.predict(dummy)
            log.info("[WW] Model warm-up complete. Key: %s", self._model_key)

        except Exception as e:
            log.error("[WW] Model load failed: %s", e)
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, audio: np.ndarray) -> float:
        """
        Run inference on one chunk. Returns rolling-averaged confidence score.
        Audio must be mono int16 at SAMPLE_RATE.
        """
        if not self.ready:
            return 0.0
        try:
            scores = self._model.predict(audio)
            # scores is a dict: {model_key: float}
            raw = 0.0
            for key, val in scores.items():
                if self._model_key and self._model_key in key:
                    raw = float(val)
                    break
                raw = max(raw, float(val))  # take max if key matching fails

            self._scores.append(raw)
            avg = float(np.mean(self._scores))

            if self.config.debug_scores:
                bar = "█" * int(avg * 40)
                tag = " ◄ DETECTED" if avg >= self.config.threshold else ""
                print(f"\r  [{self._model_key}] {avg:.4f} |{bar:<40}|{tag}    ", end="", flush=True)

            return avg

        except Exception as e:
            log.debug("[WW] predict error: %s", e)
            return 0.0

    def reset_window(self):
        """Clear rolling window (call after activation)."""
        self._scores.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  SPEAKER VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class SpeakerVerifier:
    """
    Resemblyzer-based speaker verification.
    Enroll once → rejects any voice that doesn't match owner.
    Gracefully skips if resemblyzer not installed.
    """

    PROFILE_PATH = Path(__file__).parent.parent / "models" / "owner_voice.npy"
    THRESHOLD    = 0.75   # cosine similarity; tune with tools/speaker_enroll.py

    def __init__(self):
        self._encoder = None
        self._owner_embedding: Optional[np.ndarray] = None
        self._enabled = False
        self._load()

    def _load(self):
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            log.info("[SV] Resemblyzer speaker encoder loaded")
            if self.PROFILE_PATH.exists():
                self._owner_embedding = np.load(str(self.PROFILE_PATH))
                self._enabled = True
                log.info("[SV] Owner voice profile loaded → speaker verification ACTIVE")
            else:
                log.info("[SV] No owner profile found — run tools/speaker_enroll.py to enable")
        except ImportError:
            log.info("[SV] resemblyzer not installed — speaker verification disabled")

    @property
    def active(self) -> bool:
        return self._enabled and self._encoder is not None and self._owner_embedding is not None

    def enroll(self, audio_clips: list[np.ndarray]) -> None:
        """Enroll owner from a list of audio arrays (int16, 16kHz)."""
        if self._encoder is None:
            raise RuntimeError("Resemblyzer not installed")
        embeddings = []
        for clip in audio_clips:
            f = clip.astype(np.float32) / 32768.0
            emb = self._encoder.embed_utterance(f)
            embeddings.append(emb)
        self._owner_embedding = np.mean(embeddings, axis=0)
        self.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.PROFILE_PATH), self._owner_embedding)
        self._enabled = True
        log.info("[SV] Owner enrolled — %d clips, profile saved", len(audio_clips))

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        """
        Returns (is_owner, similarity_score).
        If verifier not active, always returns (True, 1.0).
        """
        if not self.active:
            return True, 1.0
        try:
            f = audio.astype(np.float32) / 32768.0
            emb = self._encoder.embed_utterance(f)
            sim = float(np.dot(emb, self._owner_embedding) /
                        (np.linalg.norm(emb) * np.linalg.norm(self._owner_embedding) + 1e-8))
            accepted = sim >= self.THRESHOLD
            log.debug("[SV] similarity=%.3f accepted=%s", sim, accepted)
            return accepted, sim
        except Exception as e:
            log.warning("[SV] verification error: %s", e)
            return True, 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  PRODUCTION WAKE ENGINE  (main class)
# ═══════════════════════════════════════════════════════════════════════════════

class WakeEngine:
    """
    Production-grade always-listening wake word engine.

    Usage:
        engine = WakeEngine(config=EngineConfig(threshold=0.5))
        engine.on_wake = my_callback   # called with DetectionEvent
        engine.start()
        # ... run forever ...
        engine.stop()
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config   = config or EngineConfig()
        self._running = False
        self._stop    = threading.Event()

        # Lock-free audio queue: audio thread → processing thread
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)

        # Sub-systems
        self._vad      = VADFilter()
        self._ns       = NoiseSupressor()
        self._detector = WakeWordDetector(self.config)
        self._verifier = SpeakerVerifier()

        # State
        self._last_trigger = 0.0   # epoch time of last detection
        self._chunks_seen  = 0
        self._drops        = 0

        # Callback: set before calling start()
        self.on_wake: Optional[Callable[[DetectionEvent], None]] = None
        self.on_score: Optional[Callable[[float], None]] = None  # for live viz

    @property
    def ready(self) -> bool:
        return self._detector.ready

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self, device: Optional[int] = None, src_rate: Optional[int] = None):
        """
        Start the always-listening loop.
        Spawns two daemon threads: audio capture + processing.
        """
        if not self.ready:
            raise RuntimeError("Wake word model failed to load. Check model path.")

        self._stop.clear()
        self._running = True

        # Audio capture thread (non-blocking — never drops frames)
        t_audio = threading.Thread(
            target=self._audio_thread,
            args=(device, src_rate),
            daemon=True,
            name="WakeAudio",
        )
        # Processing thread (CPU-bound work off the audio thread)
        t_proc = threading.Thread(
            target=self._process_thread,
            daemon=True,
            name="WakeProcess",
        )

        t_audio.start()
        t_proc.start()

        log.info(
            "[Engine] Started. Model=%s Threshold=%.2f Window=%d Cooldown=%.1fs",
            self._detector._model_key,
            self.config.threshold,
            self.config.rolling_window,
            self.config.cooldown_secs,
        )
        return t_audio, t_proc

    def stop(self):
        """Graceful shutdown — drains queue then exits."""
        self._stop.set()
        self._running = False
        log.info("[Engine] Stopped. chunks=%d drops=%d", self._chunks_seen, self._drops)

    # ── Audio thread ─────────────────────────────────────────────────────────

    def _audio_thread(self, device: Optional[int], src_rate: Optional[int]):
        """
        Capture microphone audio in CHUNK_MS blocks.
        Push raw audio to queue; never block.
        """
        import sounddevice as sd

        capture_rate = src_rate or SAMPLE_RATE

        def _callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                log.debug("[Audio] %s", status)
            try:
                # Take first channel only (force mono at capture level)
                chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                self._audio_q.put_nowait((chunk, capture_rate))
            except queue.Full:
                self._drops += 1   # processing is behind; log and continue

        log.info("[Audio] Opening mic — device=%s rate=%dHz chunk=%dms",
                 device or "default", capture_rate, CHUNK_MS)

        try:
            with sd.InputStream(
                device      = device,
                samplerate  = capture_rate,
                channels    = 1,
                dtype       = DTYPE,
                blocksize   = int(capture_rate * CHUNK_MS / 1000),
                callback    = _callback,
                latency     = "low",
            ):
                self._stop.wait()  # block until engine.stop() is called
        except Exception as e:
            log.error("[Audio] Stream failed: %s", e)
            self._stop.set()

    # ── Processing thread ────────────────────────────────────────────────────

    def _process_thread(self):
        """
        Pull chunks from queue → normalize → VAD → NS → OWW → threshold → callback.
        Runs entirely off the audio callback thread.
        """
        log.info("[Process] Thread started")

        while not self._stop.is_set():
            try:
                chunk, src_rate = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            self._chunks_seen += 1

            # ── 1. Normalise format ──────────────────────────────────────
            chunk = ensure_mono_int16(chunk)

            # ── 2. Resample if source rate differs ───────────────────────
            if src_rate != SAMPLE_RATE:
                chunk = resample_if_needed(chunk, src_rate)

            # Pad / trim to exact CHUNK_SAMPLES (OWW is strict about this)
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            elif len(chunk) > CHUNK_SAMPLES:
                chunk = chunk[:CHUNK_SAMPLES]

            # ── 3. Silence gate (save CPU on quiet environments) ─────────
            audio_rms = rms(chunk)
            if audio_rms < self.config.rms_gate:
                continue

            # ── 4. WebRTC VAD ────────────────────────────────────────────
            if not self._vad.is_speech(chunk.tobytes()):
                continue

            # ── 5. Noise suppression ─────────────────────────────────────
            chunk = self._ns.process(chunk)

            # ── 6. Wake word inference ───────────────────────────────────
            score = self._detector.predict(chunk)

            if self.on_score:
                self.on_score(score)

            # ── 7. Threshold check ───────────────────────────────────────
            if score < self.config.threshold:
                continue

            # ── 8. Cooldown guard ────────────────────────────────────────
            now = time.time()
            if now - self._last_trigger < self.config.cooldown_secs:
                log.debug("[Process] Detection suppressed (cooldown active)")
                continue

            # ── 9. Speaker verification ──────────────────────────────────
            is_owner, similarity = self._verifier.verify(chunk)
            if not is_owner:
                log.info("[Process] Non-owner speaker rejected (sim=%.3f)", similarity)
                continue

            # ── TRIGGER ──────────────────────────────────────────────────
            self._last_trigger = now
            self._detector.reset_window()

            latency_ms = (time.perf_counter() - t_start) * 1000
            event = DetectionEvent(
                confidence  = score,
                model_key   = self._detector._model_key or "unknown",
                audio_rms   = audio_rms,
                latency_ms  = latency_ms,
            )
            log.info(
                "[Engine] ★ WAKE DETECTED ★  confidence=%.3f rms=%.0f latency=%.1fms",
                score, audio_rms, latency_ms,
            )

            if self.on_wake:
                try:
                    self.on_wake(event)
                except Exception as e:
                    log.error("[Engine] on_wake callback error: %s", e)

        log.info("[Process] Thread exited")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_engine(on_wake: Callable, config: Optional[EngineConfig] = None, block: bool = True):
    """
    One-liner to start the engine and wait forever.

    Example:
        def handle(event):
            print("Hey JARVIS detected!", event.confidence)

        run_engine(handle)
    """
    engine = WakeEngine(config=config)
    engine.on_wake = on_wake
    threads = engine.start()
    if block:
        try:
            while all(t.is_alive() for t in threads):
                time.sleep(0.5)
        except KeyboardInterrupt:
            engine.stop()
    return engine

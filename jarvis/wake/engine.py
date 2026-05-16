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
    threshold:        float = 0.60       # production floor; tune with tools/threshold_tuner.py
    rolling_window:   int   = 5          # score smoothing window
    required_hits:    int   = 3          # consecutive frames over threshold before trigger
    cooldown_secs:    float = COOLDOWN_SECS
    rms_gate:         float = RMS_SILENCE_GATE
    vad_aggressiveness: int = 3          # 3 = strictest WebRTC VAD
    vad_frame_ms:     int   = 20         # 10/20/30ms only
    vad_required_ratio: float = 0.50     # speech frames required inside each chunk
    dynamic_threshold: bool = True       # raise threshold when background scores drift up
    dynamic_margin:   float = 0.08       # margin above background score statistics
    context_ms:       int   = 1200       # rolling audio context for sklearn/ONNX features
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
    def __init__(self, aggressiveness: int = 3, frame_ms: int = 20, required_ratio: float = 0.50):
        self._vad = None
        self._frame_ms = frame_ms if frame_ms in (10, 20, 30) else 20
        self._required_ratio = max(0.0, min(1.0, required_ratio))
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            log.info("[VAD] WebRTC VAD initialised (aggressiveness=%d)", aggressiveness)
        except ImportError:
            log.warning("[VAD] webrtcvad not installed — VAD filtering disabled")

    def is_speech(self, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
        """Returns True if the audio frame is speech (or if VAD unavailable)."""
        if self._vad is None:
            return self._energy_fallback(pcm_bytes)
        # WebRTC VAD requires exactly 10/20/30ms frames
        frame_size = int(sample_rate * self._frame_ms / 1000) * 2  # * 2 for int16 bytes
        if len(pcm_bytes) < frame_size:
            return self._energy_fallback(pcm_bytes)
        try:
            total = 0
            voiced = 0
            for offset in range(0, len(pcm_bytes) - frame_size + 1, frame_size):
                total += 1
                frame = pcm_bytes[offset:offset + frame_size]
                if self._vad.is_speech(frame, sample_rate):
                    voiced += 1
            if total == 0:
                return self._energy_fallback(pcm_bytes)
            return (voiced / total) >= self._required_ratio
        except Exception:
            return self._energy_fallback(pcm_bytes)

    @staticmethod
    def _energy_fallback(pcm_bytes: bytes) -> bool:
        """Strict fallback when WebRTC VAD is unavailable or errors."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return rms(audio) >= max(RMS_SILENCE_GATE * 1.5, 120.0)


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
        self._raw_scores: deque = deque(maxlen=max(config.rolling_window, 1))
        self._background_scores: deque = deque(maxlen=250)
        self._consecutive_hits = 0
        self._context_samples = max(CHUNK_SAMPLES, int(SAMPLE_RATE * config.context_ms / 1000))
        self._audio_context = np.zeros(0, dtype=np.int16)
        self._n_features = 96 * 32
        self._onnx_session = None
        self._onnx_input = None
        self._load_model()

    def _load_model(self):
        """
        Load model. Supports three formats:
          1. sklearn .pkl  — custom trained model (your hey_jarvis_custom.pkl)
          2. sklearn .onnx — converted sklearn model (your hey_jarvis_custom.onnx)
          3. OWW built-in  — hey_jarvis_v0.1 (fallback)
        """
        model_path = self.config.model_path

        # ── Try sklearn pkl first (most reliable for custom trained models) ──
        if model_path:
            p = Path(model_path)
            # Check for matching .pkl next to whatever path was given
            pkl_path = p.with_suffix(".pkl")
            if pkl_path.exists():
                try:
                    import pickle
                    with open(pkl_path, "rb") as f:
                        self._sklearn_model = pickle.load(f)
                    self._model = "sklearn"
                    self._model_key = pkl_path.stem
                    self._n_features = self._infer_sklearn_feature_count(self._sklearn_model)
                    # Warm-up
                    dummy = np.zeros(self._n_features, dtype=np.float32)
                    self._sklearn_model.predict_proba([dummy])
                    log.info("[WW] Loaded sklearn model: %s (features=%d)",
                             pkl_path, self._n_features)
                    return
                except Exception as e:
                    log.warning("[WW] sklearn pkl load failed: %s", e)

            if p.exists() and p.suffix.lower() == ".onnx":
                try:
                    from openwakeword.model import Model as OWWModel
                    self._model = OWWModel(
                        wakeword_models=[str(p)],
                        inference_framework="onnx",
                        enable_speex_noise_suppression=False,
                    )
                    self._model_key = p.stem
                    self._model.predict(np.zeros(CHUNK_SAMPLES, dtype=np.int16))
                    log.info("[WW] Loaded custom OpenWakeWord ONNX model: %s", p)
                    return
                except Exception as e:
                    log.warning("[WW] custom OpenWakeWord ONNX load failed: %s", e)

                try:
                    import onnxruntime as ort
                    self._onnx_session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                    self._onnx_input = self._onnx_session.get_inputs()[0].name
                    self._n_features = self._infer_onnx_feature_count(self._onnx_session)
                    self._model = "sklearn_onnx"
                    self._model_key = p.stem
                    dummy = np.zeros((1, self._n_features), dtype=np.float32)
                    self._onnx_session.run(None, {self._onnx_input: dummy})
                    log.info("[WW] Loaded sklearn ONNX model: %s (features=%d)", p, self._n_features)
                    return
                except Exception as e:
                    log.warning("[WW] sklearn ONNX load failed: %s", e)

        # ── Try OWW built-in (fallback) ──────────────────────────────────────
        try:
            from openwakeword.model import Model as OWWModel
            self._model = OWWModel(
                wakeword_models=["hey_jarvis_v0.1"],
                inference_framework="onnx",
                enable_speex_noise_suppression=False,
            )
            self._model_key = "hey_jarvis_v0.1"
            dummy = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            self._model.predict(dummy)
            log.info("[WW] Loaded built-in OWW model: hey_jarvis_v0.1")
        except Exception as e:
            log.error("[WW] All model loads failed: %s", e)
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def _infer_sklearn_feature_count(model) -> int:
        """Read feature width from a sklearn pipeline/model."""
        for attr_owner in (
            getattr(model, "named_steps", {}).get("scaler") if hasattr(model, "named_steps") else None,
            getattr(model, "named_steps", {}).get("mlp") if hasattr(model, "named_steps") else None,
            model,
        ):
            if attr_owner is None:
                continue
            n_features = getattr(attr_owner, "n_features_in_", None)
            if n_features:
                return int(n_features)
        return 96 * 32

    @staticmethod
    def _infer_onnx_feature_count(session) -> int:
        shape = session.get_inputs()[0].shape
        if len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])
        return 96 * 32

    def _push_context(self, audio: np.ndarray) -> np.ndarray:
        self._audio_context = np.concatenate((self._audio_context, audio.astype(np.int16)))
        if len(self._audio_context) > self._context_samples:
            self._audio_context = self._audio_context[-self._context_samples:]
        return self._audio_context

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel features from a 1280-sample int16 chunk.
        Must match the feature extraction used during training.
        """
        n_features = getattr(self, "_n_features", 40)
        audio_f = audio.astype(np.float32) / 32768.0
        frame_size = 400
        hop_size   = 160
        features   = []
        for start in range(0, len(audio_f) - frame_size, hop_size):
            frame = audio_f[start:start + frame_size]
            # Log energy in frequency bands
            fft    = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            n_bins = len(fft)
            # Split into n_features mel-like bands
            band_size = max(1, n_bins // n_features)
            bands = [
                np.log1p(np.mean(fft[i*band_size:(i+1)*band_size]))
                for i in range(n_features)
            ]
            features.append(bands)
        if not features:
            return np.zeros(n_features, dtype=np.float32)
        # Aggregate: mean + std across frames → 2*n_features, trim to n_features
        arr  = np.array(features, dtype=np.float32)
        mean = np.mean(arr, axis=0)
        return mean[:n_features].astype(np.float32)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract the same 96x32 log-mel window used by wake/train.py.
        This overrides the older single-chunk FFT extractor above.
        """
        target_features = getattr(self, "_n_features", 96 * 32)
        n_mels = 32
        n_frames = max(1, target_features // n_mels)
        y = audio.astype(np.float32) / 32768.0

        if len(y) < SAMPLE_RATE:
            y = np.pad(y, (SAMPLE_RATE - len(y), 0))
        else:
            y = y[-SAMPLE_RATE:]

        try:
            import librosa
            y = librosa.effects.preemphasis(y, coef=0.97)
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=SAMPLE_RATE,
                n_mels=n_mels,
                hop_length=160,
                win_length=400,
                power=2.0,
            )
            feat = librosa.power_to_db(mel, ref=np.max).T
        except Exception:
            feat = self._fallback_logmel(y, n_mels=n_mels)

        if len(feat) < n_frames:
            feat = np.pad(feat, ((n_frames - len(feat), 0), (0, 0)), constant_values=-80.0)
        feat = feat[-n_frames:].astype(np.float32)
        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)
        flat = feat.reshape(-1)
        if len(flat) < target_features:
            flat = np.pad(flat, (0, target_features - len(flat)))
        return flat[:target_features].astype(np.float32)

    @staticmethod
    def _fallback_logmel(y: np.ndarray, n_mels: int = 32) -> np.ndarray:
        frame_size = 400
        hop_size = 160
        rows = []
        for start in range(0, max(1, len(y) - frame_size + 1), hop_size):
            frame = y[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size))) ** 2
            band_edges = np.linspace(0, len(spectrum), n_mels + 1, dtype=int)
            bands = []
            for idx in range(n_mels):
                lo, hi = band_edges[idx], max(band_edges[idx + 1], band_edges[idx] + 1)
                bands.append(10.0 * np.log10(np.mean(spectrum[lo:hi]) + 1e-10))
            rows.append(bands)
        return np.array(rows, dtype=np.float32)

    def predict(self, audio: np.ndarray) -> float:
        """
        Run inference on one chunk. Returns rolling-averaged confidence score.
        Audio must be mono int16 at SAMPLE_RATE.
        """
        if not self.ready:
            return 0.0
        try:
            # ── sklearn pkl path ─────────────────────────────────────────
            if self._model == "sklearn":
                context = self._push_context(audio)
                feats = self._extract_features(context).reshape(1, -1)
                proba = self._sklearn_model.predict_proba(feats)[0]
                # proba[1] = probability of class 1 (wake word)
                raw = float(proba[1]) if len(proba) > 1 else float(proba[0])

            elif self._model == "sklearn_onnx":
                context = self._push_context(audio)
                feats = self._extract_features(context).reshape(1, -1).astype(np.float32)
                outputs = self._onnx_session.run(None, {self._onnx_input: feats})
                raw = self._extract_onnx_probability(outputs)

            # ── OWW built-in path ────────────────────────────────────────
            else:
                scores = self._model.predict(audio)
                raw = 0.0
                for key, val in scores.items():
                    if self._model_key and self._model_key in key:
                        raw = float(val)
                        break
                    raw = max(raw, float(val))

            self._raw_scores.append(raw)
            if raw < self.config.threshold * 0.75:
                self._background_scores.append(raw)
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

    @staticmethod
    def _extract_onnx_probability(outputs) -> float:
        """Handle sklearn-onnx probability outputs across converter versions."""
        for output in outputs:
            if isinstance(output, list) and output and isinstance(output[0], dict):
                probs = output[0]
                return float(probs.get(1, probs.get("1", max(probs.values()))))
            arr = np.asarray(output)
            if arr.ndim == 2 and arr.shape[1] > 1 and np.issubdtype(arr.dtype, np.number):
                return float(arr[0, 1])
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                return float(arr.reshape(-1)[0])
        return 0.0

    def current_threshold(self) -> float:
        """Adaptive threshold: never below config.threshold, higher in noisy rooms."""
        base = float(self.config.threshold)
        if not self.config.dynamic_threshold or len(self._background_scores) < 25:
            return base
        bg = np.array(self._background_scores, dtype=np.float32)
        adaptive = float(np.mean(bg) + 3.0 * np.std(bg) + self.config.dynamic_margin)
        return min(0.95, max(base, adaptive))

    def is_confirmed(self, score: float) -> bool:
        """Require a sliding average plus N consecutive hits above threshold."""
        threshold = self.current_threshold()
        if score >= threshold:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
        return self._consecutive_hits >= max(1, self.config.required_hits)

    def reset_window(self):
        """Clear rolling window (call after activation)."""
        self._scores.clear()
        self._raw_scores.clear()
        self._consecutive_hits = 0


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
        self._vad      = VADFilter(
            aggressiveness=self.config.vad_aggressiveness,
            frame_ms=self.config.vad_frame_ms,
            required_ratio=self.config.vad_required_ratio,
        )
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
                self._detector.reset_window()
                continue

            # ── 4. WebRTC VAD ────────────────────────────────────────────
            if not self._vad.is_speech(chunk.tobytes()):
                self._detector.reset_window()
                continue

            # ── 5. Noise suppression ─────────────────────────────────────
            chunk = self._ns.process(chunk)

            # ── 6. Wake word inference ───────────────────────────────────
            score = self._detector.predict(chunk)

            if self.on_score:
                self.on_score(score)

            # ── 7. Threshold check ───────────────────────────────────────
            if not self._detector.is_confirmed(score):
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

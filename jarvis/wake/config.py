"""
wake/config.py — JARVIS Wake Word System Configuration
=======================================================
Single source of truth for all tunable parameters.
Edit this file to adjust sensitivity, performance, and behavior.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
WAKE_DIR      = Path(__file__).parent
JARVIS_DIR    = WAKE_DIR.parent
DATA_DIR      = JARVIS_DIR / "data"
MODELS_DIR    = WAKE_DIR / "models"
DATASET_DIR   = WAKE_DIR / "dataset"
EMBEDDINGS_DIR = DATA_DIR / "speaker_embeddings"

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000          # Hz — required by OpenWakeWord + Whisper
CHANNELS      = 1              # Mono
DTYPE         = "int16"
CHUNK_MS      = 80             # ms per chunk — 80ms is openwakeword's sweet spot
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)   # = 1280 samples

# ── Wake Word Detection ──────────────────────────────────────────────────────
CUSTOM_WAKE_MODEL = MODELS_DIR / "hey_jarvis_custom.onnx"
WAKE_MODELS = [
    str(CUSTOM_WAKE_MODEL) if CUSTOM_WAKE_MODEL.exists() else "alexa",
]
WAKE_THRESHOLD          = 0.60   # 0.0–1.0 — higher = fewer false positives
WAKE_COOLDOWN_SECONDS   = 2.0    # minimum gap between detections
WAKE_ROLLING_WINDOW     = 3      # confirm over N consecutive chunks (reduces false +)

# ── Voice Activity Detection ─────────────────────────────────────────────────
VAD_AGGRESSIVENESS      = 3      # 0=least, 3=most aggressive filtering
VAD_FRAME_MS            = 20     # WebRTC VAD frame size (10, 20, or 30 ms only)
VAD_SILENCE_THRESHOLD_S = 1.2    # stop recording after this many seconds of silence
VAD_MIN_SPEECH_S        = 0.3    # discard if total speech shorter than this

# ── Speaker Verification ─────────────────────────────────────────────────────
SPEAKER_VERIFY_ENABLED  = False  # Set True after enrolling your voice
SPEAKER_SIMILARITY_MIN  = 0.72   # cosine similarity 0–1; tune experimentally
SPEAKER_ENROLL_SECONDS  = 5.0    # seconds of audio per enrollment sample
SPEAKER_ENROLL_COUNT    = 5      # number of enrollment samples to average

# ── STT (Faster-Whisper) ─────────────────────────────────────────────────────
WHISPER_MODEL           = "tiny.en"   # tiny.en=~39MB, base.en=~142MB
WHISPER_DEVICE          = "cpu"
WHISPER_COMPUTE_TYPE    = "int8"      # int8 = fastest on CPU
WHISPER_BEAM_SIZE       = 3
WHISPER_MAX_DURATION_S  = 10.0        # hard cap on capture window

# ── Noise Suppression ────────────────────────────────────────────────────────
NOISE_SUPPRESS_ENABLED  = True
NOISE_RNNOISE_ENABLED   = False       # True if rnnoise Python binding installed

# ── Reliability ──────────────────────────────────────────────────────────────
WATCHDOG_INTERVAL_S     = 10.0        # restart thread if it hasn't produced a heartbeat
MAX_RESTART_ATTEMPTS    = 5           # give up after this many consecutive crashes
STREAM_TIMEOUT_S        = 30.0        # SSE subscriber heartbeat interval

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DETECTIONS          = True        # log every detection with score
LOG_AUDIO_CHUNKS        = False       # verbose — only for debugging

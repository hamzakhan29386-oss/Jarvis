"""
wake/noise.py — Noise Suppression Pipeline
============================================
Two-stage filtering:
    Stage 1: WebRTC VAD (webrtcvad) — frame-level speech detection
    Stage 2: RNNoise (optional) — spectral noise suppression

WebRTC VAD classifies each 20ms frame as speech or silence.
RNNoise suppresses stationary background noise (fans, hum, etc.)

Requirements:
    pip install webrtcvad
    pip install rnnoise-python   # optional; falls back gracefully
"""

import logging
import struct
import numpy as np
from typing import Tuple

from .config import (
    SAMPLE_RATE, VAD_AGGRESSIVENESS, VAD_FRAME_MS,
    NOISE_SUPPRESS_ENABLED, NOISE_RNNOISE_ENABLED,
)

log = logging.getLogger("jarvis.wake.noise")


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBRTC VAD
# ═══════════════════════════════════════════════════════════════════════════════

class WebRTCVAD:
    """
    Thin wrapper around webrtcvad for per-frame speech detection.
    Classifies 10/20/30ms frames as speech (True) or silence (False).
    """

    VALID_FRAME_MS = (10, 20, 30)

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS,
                 frame_ms: int = VAD_FRAME_MS):
        assert frame_ms in self.VALID_FRAME_MS, \
            f"frame_ms must be one of {self.VALID_FRAME_MS}"
        self._aggressiveness = aggressiveness
        self._frame_ms = frame_ms
        self._frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
        self._vad = None
        self._available = False
        self._init()

    def _init(self):
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self._aggressiveness)
            self._available = True
            log.info(f"[VAD] WebRTC VAD ready (aggressiveness={self._aggressiveness})")
        except ImportError:
            log.warning(
                "[VAD] webrtcvad not installed — VAD disabled. "
                "Run: pip install webrtcvad"
            )

    @property
    def available(self) -> bool:
        return self._available

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Classify a chunk of audio as speech or silence.

        Args:
            pcm_bytes: 16-bit mono PCM audio bytes. Must be exactly
                       frame_ms milliseconds of audio.

        Returns:
            True if speech detected, False if silence.
        """
        if not self._available:
            return True   # assume speech when VAD unavailable

        # Pad or trim to exact frame length if needed
        expected = self._frame_samples * 2  # 2 bytes per int16 sample
        if len(pcm_bytes) != expected:
            if len(pcm_bytes) < expected:
                pcm_bytes = pcm_bytes.ljust(expected, b'\x00')
            else:
                pcm_bytes = pcm_bytes[:expected]

        try:
            return self._vad.is_speech(pcm_bytes, SAMPLE_RATE)
        except Exception as e:
            log.debug(f"[VAD] is_speech error: {e}")
            return True

    def filter_chunk(self, chunk_bytes: bytes) -> Tuple[bool, float]:
        """
        Run VAD over a larger chunk by splitting into frames.

        Args:
            chunk_bytes: Arbitrary-length 16-bit mono PCM.

        Returns:
            (has_speech: bool, speech_ratio: float 0.0–1.0)
        """
        if not self._available:
            return True, 1.0

        frame_bytes = self._frame_samples * 2
        total_frames = 0
        speech_frames = 0

        for offset in range(0, len(chunk_bytes) - frame_bytes + 1, frame_bytes):
            frame = chunk_bytes[offset:offset + frame_bytes]
            total_frames += 1
            if self.is_speech(frame):
                speech_frames += 1

        if total_frames == 0:
            return False, 0.0

        ratio = speech_frames / total_frames
        return ratio > 0.3, ratio   # >30% speech frames → classify as speech


# ═══════════════════════════════════════════════════════════════════════════════
#  RNNOISE (optional)
# ═══════════════════════════════════════════════════════════════════════════════

class RNNoiseSuppressor:
    """
    Spectral noise suppression via RNNoise.
    Removes stationary background noise (fans, HVAC, keyboard hum).

    Installation:
        pip install rnnoise-python   # unofficial binding
        # OR
        pip install noisereduce      # fallback spectral subtraction
    """

    def __init__(self):
        self._denoiser = None
        self._mode = None
        self._init()

    def _init(self):
        if not NOISE_RNNOISE_ENABLED:
            log.info("[Noise] RNNoise disabled in config.")
            return

        # Try rnnoise-python first
        try:
            import rnnoise
            self._denoiser = rnnoise.RNNoise()
            self._mode = "rnnoise"
            log.info("[Noise] RNNoise suppressor ready")
            return
        except ImportError:
            pass

        # Fallback: noisereduce (spectral subtraction)
        try:
            import noisereduce  # noqa
            self._mode = "noisereduce"
            log.info("[Noise] noisereduce fallback suppressor ready")
            return
        except ImportError:
            pass

        log.warning(
            "[Noise] No noise suppression library found. "
            "Run: pip install rnnoise-python  OR  pip install noisereduce"
        )

    @property
    def available(self) -> bool:
        return self._mode is not None

    def suppress(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Apply noise suppression to a numpy audio array.

        Args:
            audio_np: int16 numpy array.

        Returns:
            Denoised int16 numpy array.
        """
        if not self.available:
            return audio_np

        try:
            if self._mode == "rnnoise":
                return self._suppress_rnnoise(audio_np)
            elif self._mode == "noisereduce":
                return self._suppress_noisereduce(audio_np)
        except Exception as e:
            log.debug(f"[Noise] Suppression error: {e}")

        return audio_np

    def _suppress_rnnoise(self, audio_np: np.ndarray) -> np.ndarray:
        """RNNoise operates on 480-sample frames at 48kHz, resample if needed."""
        import rnnoise
        # rnnoise expects float32 in range [-32768, 32767]
        audio_f = audio_np.astype(np.float32)
        # Process in 480-sample frames
        FRAME = 480
        result = []
        for i in range(0, len(audio_f) - FRAME + 1, FRAME):
            frame = audio_f[i:i + FRAME]
            denoised, _ = self._denoiser.process_frame(frame)
            result.append(denoised)
        if result:
            out = np.concatenate(result).astype(np.int16)
            # Pad/trim to original length
            if len(out) < len(audio_np):
                out = np.pad(out, (0, len(audio_np) - len(out)))
            return out[:len(audio_np)]
        return audio_np

    def _suppress_noisereduce(self, audio_np: np.ndarray) -> np.ndarray:
        import noisereduce as nr
        audio_f = audio_np.astype(np.float32) / 32768.0
        denoised = nr.reduce_noise(y=audio_f, sr=SAMPLE_RATE, stationary=False)
        return (denoised * 32767).clip(-32768, 32767).astype(np.int16)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED NOISE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class NoisePipeline:
    """
    Combined noise suppression + VAD pipeline.
    Used by the wake word service to preprocess each audio chunk.
    """

    def __init__(self):
        self.vad = WebRTCVAD()
        self.rnnoise = RNNoiseSuppressor() if NOISE_SUPPRESS_ENABLED else None

    def process(self, chunk_bytes: bytes) -> Tuple[bytes, bool, float]:
        """
        Full preprocessing pipeline for one audio chunk.

        Args:
            chunk_bytes: Raw 16-bit mono PCM bytes.

        Returns:
            (processed_bytes, has_speech, speech_ratio)
        """
        audio_np = np.frombuffer(chunk_bytes, dtype=np.int16).copy()

        # Stage 1: Spectral noise suppression
        if self.rnnoise and self.rnnoise.available:
            audio_np = self.rnnoise.suppress(audio_np)

        processed_bytes = audio_np.tobytes()

        # Stage 2: VAD
        has_speech, ratio = self.vad.filter_chunk(processed_bytes)

        return processed_bytes, has_speech, ratio

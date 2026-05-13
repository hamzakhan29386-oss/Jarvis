"""Real-time audio resampling helpers for native microphone capture."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample

log = logging.getLogger("jarvis.voice.resampler")


@dataclass(frozen=True)
class ResampleConfig:
    source_rate: int = 48000
    target_rate: int = 16000


class AudioResampler:
    """Resample mono audio blocks with scipy.signal.resample."""

    def __init__(self, config: ResampleConfig | None = None):
        self.config = config or ResampleConfig()

    @property
    def source_rate(self) -> int:
        return self.config.source_rate

    @property
    def target_rate(self) -> int:
        return self.config.target_rate

    def process(self, audio: np.ndarray, source_rate: int | None = None) -> np.ndarray:
        src_rate = int(source_rate or self.source_rate)
        if src_rate == self.target_rate:
            return audio.astype(np.float32, copy=False)
        if audio.size == 0:
            return np.empty(0, dtype=np.float32)

        target_len = max(1, int(round(audio.shape[0] * self.target_rate / src_rate)))
        t0 = __import__("time").perf_counter()
        out = resample(audio.astype(np.float32, copy=False), target_len)
        latency_ms = (__import__("time").perf_counter() - t0) * 1000
        log.debug(
            "[Resampler] %dHz -> %dHz | in=%d out=%d latency=%.2fms",
            src_rate,
            self.target_rate,
            audio.shape[0],
            out.shape[0],
            latency_ms,
        )
        return out.astype(np.float32, copy=False)


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio
    clipped = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)

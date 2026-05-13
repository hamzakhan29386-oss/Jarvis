"""Asynchronous native wake listener built on Python-owned microphone input."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from wake.config import CHUNK_SAMPLES, SAMPLE_RATE, SPEAKER_VERIFY_ENABLED
from wake.detector import WakeWordEngine
from wake.noise import NoisePipeline
from wake.speaker import SpeakerVerifier

from .audio_input import AudioInputConfig, NativeAudioInput
from .resampler import AudioResampler, float32_to_int16

log = logging.getLogger("jarvis.voice.wake_listener")


class NativeWakeListener:
    """Continuous wake detector with callback-based audio capture and worker processing."""

    def __init__(self):
        self.audio = NativeAudioInput(AudioInputConfig())
        self.resampler = AudioResampler()
        self.noise = NoisePipeline()
        self.detector = WakeWordEngine()
        self.speaker = SpeakerVerifier()
        self.on_detected = None
        self.on_error = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._buffer = np.empty(0, dtype=np.int16)
        self._chunks_processed = 0
        self._last_latency_ms = 0.0
        self._last_detection: dict | None = None

    @property
    def available(self) -> bool:
        return self.detector.available

    def start(self) -> None:
        if self._running:
            return
        if not self.detector.available:
            raise RuntimeError("Wake detector unavailable")
        self._stop.clear()
        self.audio.start()
        self._thread = threading.Thread(target=self._process_loop, daemon=True, name="JARVISNativeWakeProcess")
        self._thread.start()
        self._running = True
        log.info("[WakeListener] Started native listener | model_status=%s", self.detector.get_status())

    def stop(self) -> None:
        self._stop.set()
        self.audio.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._running = False
        log.info("[WakeListener] Stopped | chunks=%d status=%s", self._chunks_processed, self.audio.get_status())

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.audio.read(timeout=0.5)
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                resampled = self.resampler.process(frame.data, frame.samplerate)
                pcm16 = float32_to_int16(resampled)
                self._buffer = np.concatenate((self._buffer, pcm16))

                while self._buffer.size >= CHUNK_SAMPLES:
                    chunk = self._buffer[:CHUNK_SAMPLES]
                    self._buffer = self._buffer[CHUNK_SAMPLES:]
                    self._process_chunk(chunk, t0)
            except Exception as exc:
                log.exception("[WakeListener] Processing error")
                if self.on_error:
                    self.on_error(exc)

    def _process_chunk(self, chunk: np.ndarray, t0: float) -> None:
        self._chunks_processed += 1
        processed_bytes, has_speech, _ = self.noise.process(chunk.tobytes())
        if not has_speech:
            return
        processed = np.frombuffer(processed_bytes, dtype=np.int16)
        detected, keyword, score = self.detector.process_chunk(processed)
        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        if not detected:
            if self._chunks_processed % 100 == 0:
                log.debug(
                    "[WakeListener] chunks=%d queue=%d drops=%d latency=%.1fms",
                    self._chunks_processed,
                    self.audio.queue.qsize(),
                    self.audio.stats.frames_dropped,
                    self._last_latency_ms,
                )
            return

        if SPEAKER_VERIFY_ENABLED and self.speaker and self.speaker.enrolled:
            ok, sim = self.speaker.verify(processed)
            if not ok:
                log.info("[WakeListener] Speaker rejected similarity=%.3f", sim)
                return
            log.info("[WakeListener] Speaker verified similarity=%.3f", sim)

        event = {
            "keyword": keyword,
            "score": round(float(score), 3),
            "latency_ms": round(self._last_latency_ms, 1),
            "queue_size": self.audio.queue.qsize(),
            "dropped_frames": self.audio.stats.frames_dropped,
            "sample_rate": SAMPLE_RATE,
        }
        self._last_detection = event
        log.info(
            "[WakeListener] Wake detected | keyword=%s score=%.3f latency=%.1fms queue=%d drops=%d",
            keyword,
            score,
            self._last_latency_ms,
            self.audio.queue.qsize(),
            self.audio.stats.frames_dropped,
        )
        if self.on_detected:
            self.on_detected(event)

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "available": self.detector.available,
            "chunks_processed": self._chunks_processed,
            "last_latency_ms": round(self._last_latency_ms, 1),
            "last_detection": self._last_detection,
            "detector": self.detector.get_status(),
            "speaker": self.speaker.get_status(),
            "audio": self.audio.get_status(),
            "resampler": {
                "source_rate": self.resampler.source_rate,
                "target_rate": self.resampler.target_rate,
            },
        }

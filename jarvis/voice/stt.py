"""Post-wake STT using native microphone capture and future streaming hooks."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave

import numpy as np

from wake.config import (
    SAMPLE_RATE,
    VAD_MIN_SPEECH_S,
    VAD_SILENCE_THRESHOLD_S,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MAX_DURATION_S,
    WHISPER_MODEL,
)
from wake.noise import NoisePipeline

from .audio_multiplexer import get_audio_multiplexer
from .resampler import AudioResampler, float32_to_int16

log = logging.getLogger("jarvis.voice.stt")


class NativeSTT:
    """Capture a command from the same native audio path and transcribe it."""

    def __init__(self):
        self._model = None
        self._available = False
        self._lock = threading.Lock()
        self._noise = NoisePipeline()
        self._resampler = AudioResampler()
        self._init_model()

    @property
    def available(self) -> bool:
        return self._available

    def _init_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            log.info("[STT] Loading Faster-Whisper model=%s device=%s", WHISPER_MODEL, WHISPER_DEVICE)
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            self._available = True
            log.info("[STT] Faster-Whisper ready")
        except ImportError:
            log.error("[STT] faster-whisper not installed")
        except Exception as exc:
            log.error("[STT] Init failed: %s", exc)

    def capture_and_transcribe(
        self,
        max_duration_s: float = WHISPER_MAX_DURATION_S,
        silence_timeout_s: float = VAD_SILENCE_THRESHOLD_S,
        min_speech_s: float = VAD_MIN_SPEECH_S,
        status_callback=None,
    ) -> str:
        if not self._available:
            return ""
        if status_callback:
            status_callback("listening")

        audio = get_audio_multiplexer()
        started_here = not audio.is_running()
        audio.start()
        subscription = audio.subscribe("stt", maxsize=96)
        frames: list[np.ndarray] = []
        speech_samples = 0
        silence_start: float | None = None
        start = time.monotonic()

        try:
            log.info("[STT] Native multiplexer capture started for command")
            while time.monotonic() - start < max_duration_s:
                frame = subscription.read(timeout=0.25)
                if frame is None:
                    continue
                audio16 = self._resampler.process(frame.data, frame.samplerate)
                pcm16 = float32_to_int16(audio16)
                processed, has_speech, _ = self._noise.process(pcm16.tobytes())
                chunk = np.frombuffer(processed, dtype=np.int16)
                frames.append(chunk)

                if has_speech:
                    speech_samples += len(chunk)
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= silence_timeout_s and speech_samples > 0:
                        break
        finally:
            audio.unsubscribe(subscription)
            if started_here:
                audio.stop()

        if not frames:
            return ""
        audio = np.concatenate(frames)
        speech_seconds = speech_samples / SAMPLE_RATE
        if speech_seconds < min_speech_s:
            log.info("[STT] Too little speech %.2fs < %.2fs", speech_seconds, min_speech_s)
            return ""
        return self._transcribe(audio)

    def _transcribe(self, audio_np: np.ndarray) -> str:
        tmp_path = os.path.join(tempfile.gettempdir(), "jarvis_native_stt.wav")
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            with self._lock:
                t0 = time.perf_counter()
                segments, _ = self._model.transcribe(
                    tmp_path,
                    beam_size=WHISPER_BEAM_SIZE,
                    language="en",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                text = " ".join(s.text for s in segments).strip()
                log.info("[STT] Transcript latency=%.1fms text=%r", (time.perf_counter() - t0) * 1000, text)
                return text
        except Exception as exc:
            log.error("[STT] Transcription error: %s", exc)
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def get_status(self) -> dict:
        return {
            "available": self._available,
            "model": WHISPER_MODEL,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
        }

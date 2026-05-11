"""
wake/transcriber.py — Post-Wake STT via Faster-Whisper
=========================================================
Captures audio after wake word detection and transcribes it.
Optimized for minimal latency on CPU with int8 quantization.

Pipeline:
    1. Open mic stream
    2. Accumulate speech frames via VAD
    3. Stop on silence or hard timeout
    4. Transcribe with faster-whisper
    5. Return text

Requirements:
    pip install faster-whisper sounddevice
"""

import logging
import threading
import time
import wave
import tempfile
import os
from typing import Optional

import numpy as np

from .config import (
    SAMPLE_RATE, CHUNK_SAMPLES,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE,
    VAD_SILENCE_THRESHOLD_S, VAD_MIN_SPEECH_S, WHISPER_MAX_DURATION_S,
)
from .noise import NoisePipeline

log = logging.getLogger("jarvis.wake.transcriber")


class PostWakeTranscriber:
    """
    Record user command after wake word, then transcribe with Faster-Whisper.
    Thread-safe; multiple callers will serialize via a lock.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._noise = NoisePipeline()
        self._available = False
        self._init()

    def _init(self):
        try:
            from faster_whisper import WhisperModel
            log.info(f"[STT] Loading Faster-Whisper ({WHISPER_MODEL})...")
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            self._available = True
            log.info("[STT] Faster-Whisper ready")
        except ImportError:
            log.error(
                "[STT] faster-whisper not installed. "
                "Run: pip install faster-whisper"
            )
        except Exception as e:
            log.error(f"[STT] Init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def capture_and_transcribe(
        self,
        max_duration_s: float = WHISPER_MAX_DURATION_S,
        silence_timeout_s: float = VAD_SILENCE_THRESHOLD_S,
        min_speech_s: float = VAD_MIN_SPEECH_S,
        status_callback=None,
    ) -> str:
        """
        Open the mic, capture one command, and return transcribed text.

        Args:
            max_duration_s: Hard cap on recording length.
            silence_timeout_s: Stop after N seconds of silence.
            min_speech_s: Discard if less than N seconds of speech captured.
            status_callback: Optional callable(event_str) for UI feedback.

        Returns:
            Transcribed string, or "" on failure/silence.
        """
        if not self._available:
            return ""

        try:
            import sounddevice as sd
        except ImportError:
            log.error("[STT] sounddevice not installed")
            return ""

        if status_callback:
            status_callback("listening")

        frames = []
        silence_start: Optional[float] = None
        speech_frames = 0
        start_time = time.monotonic()
        stop_event = threading.Event()

        def audio_callback(indata, frame_count, time_info, status):
            nonlocal silence_start, speech_frames
            raw = indata.flatten()
            processed, has_speech, _ = self._noise.process(raw.tobytes())
            frames.append(np.frombuffer(processed, dtype=np.int16))

            if has_speech:
                speech_frames += len(raw)
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= silence_timeout_s:
                    stop_event.set()

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        )

        with stream:
            log.info("[STT] Mic open — capturing command...")
            while not stop_event.is_set():
                if time.monotonic() - start_time >= max_duration_s:
                    log.info("[STT] Max duration reached")
                    break
                time.sleep(0.02)

        if not frames:
            log.info("[STT] No audio captured")
            return ""

        total_audio = np.concatenate(frames)
        speech_seconds = speech_frames / SAMPLE_RATE

        if speech_seconds < min_speech_s:
            log.info(f"[STT] Too short ({speech_seconds:.2f}s < {min_speech_s}s) — discarding")
            return ""

        text = self._transcribe(total_audio)
        log.info(f"[STT] Transcript: '{text}'")
        return text

    def _transcribe(self, audio_np: np.ndarray) -> str:
        """Write audio to a temp WAV, transcribe it, return text."""
        if not self._available:
            return ""

        tmp_path = os.path.join(tempfile.gettempdir(), "jarvis_wake_stt.wav")
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            with self._lock:
                segments, _ = self._model.transcribe(
                    tmp_path,
                    beam_size=WHISPER_BEAM_SIZE,
                    language="en",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                return " ".join(s.text for s in segments).strip()

        except Exception as e:
            log.error(f"[STT] Transcription error: {e}")
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

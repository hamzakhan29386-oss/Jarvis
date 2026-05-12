"""
voice.py — JARVIS Voice Engine (Real-Time Streaming Pipeline)
==============================================================
UPGRADED PIPELINE:
  OLD: listen → wait → full AI response → TTS → play
  NEW: listen → stream tokens → speak sentence-by-sentence immediately

Key additions:
  • StreamingSpeaker  — buffers tokens, speaks each sentence as it completes
  • PiperTTS          — fast neural TTS (optional, ~50MB model, natural voice)
  • Tighter VAD       — 0.5s silence timeout (was 1.5s)
  • stream_speak()    — VoiceEngine method for the new pipeline

Stack:
    STT:       faster-whisper (tiny.en, ~39MB, int8)
    TTS:       Piper (neural, offline) with pyttsx3 fallback
    VAD:       silero-vad (voice activity detection)
    Wake Word: openwakeword ("hey jarvis")
"""

import os
import re
import io
import sys
import wave
import time
import queue
import logging
import subprocess
import threading
import tempfile
from pathlib import Path

log = logging.getLogger("jarvis.voice")


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMING SPEAKER — core of the new real-time pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingSpeaker:
    """
    Sentence-level streaming TTS.

    Feed tokens one at a time. The speaker detects sentence boundaries
    and queues each completed sentence for playback immediately —
    so JARVIS starts speaking before the AI finishes generating.

    Usage:
        speaker = StreamingSpeaker(tts_engine)
        for token in think_stream(prompt):
            speaker.feed_token(token)
        speaker.flush()          # speak any trailing text
        speaker.wait_done()      # block until all audio finishes
    """

    # Sentence-ending pattern: . ! ? followed by space, newline, or end of string
    _SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])$')
    # Also split on newlines that separate paragraphs
    _PARA_SPLIT   = re.compile(r'\n{2,}')

    def __init__(self, tts_engine):
        self._tts    = tts_engine
        self._buffer = ""
        self._queue  = queue.Queue(maxsize=16)
        self._done   = threading.Event()
        self._active = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="tts-streamer"
        )
        self._worker.start()

    # ── Internal TTS worker ──────────────────────────────────────────────

    def _worker_loop(self):
        """Background thread: dequeue sentences and speak them one by one."""
        while self._active:
            try:
                sentence = self._queue.get(timeout=1.0)
                if sentence is None:      # poison pill — shut down
                    break
                if sentence.strip():
                    self._tts.speak(sentence.strip())
                self._queue.task_done()
            except queue.Empty:
                continue
        self._done.set()

    def _enqueue(self, sentence: str):
        """Push a sentence to the playback queue (non-blocking)."""
        if sentence.strip():
            try:
                self._queue.put_nowait(sentence.strip())
            except queue.Full:
                # Queue full: block briefly rather than drop audio
                try:
                    self._queue.put(sentence.strip(), timeout=5.0)
                except queue.Full:
                    log.warning("[StreamingSpeaker] Queue full, dropping sentence")

    # ── Public API ───────────────────────────────────────────────────────

    def feed_token(self, token: str):
        """
        Add a token to the buffer. Speak any completed sentences immediately.
        Call this for every token from think_stream().
        """
        self._buffer += token

        # Check for sentence boundaries
        parts = self._SENTENCE_END.split(self._buffer)
        if len(parts) > 1:
            # parts[-1] is the incomplete tail; everything before is a sentence
            for sentence in parts[:-1]:
                self._enqueue(sentence)
            self._buffer = parts[-1]

        # Also handle paragraph breaks
        paras = self._PARA_SPLIT.split(self._buffer)
        if len(paras) > 1:
            for para in paras[:-1]:
                if para.strip():
                    self._enqueue(para)
            self._buffer = paras[-1]

    def feed_tokens(self, token_generator):
        """
        Convenience method: iterate a token generator and feed each token.
        Automatically flushes when the generator is exhausted.
        """
        for token in token_generator:
            self.feed_token(token)
        self.flush()

    def flush(self):
        """Speak any text remaining in the buffer (call after stream ends)."""
        if self._buffer.strip():
            self._enqueue(self._buffer)
            self._buffer = ""

    def wait_done(self, timeout: float = 60.0):
        """Block until the TTS queue is empty and all audio has played."""
        self._queue.join()

    def stop(self):
        """Interrupt playback and shut down the worker."""
        self._active = False
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put(None)   # poison pill


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPER TTS — fast neural voice (optional)
# ═══════════════════════════════════════════════════════════════════════════════

class PiperTTS:
    """
    Neural TTS using Piper — natural-sounding, offline, ~50ms latency.

    Setup (one-time):
        pip install piper-tts
        python -c "from piper import download; download('en_US-lessac-medium')"

    Falls back to Pyttsx3TTS if Piper is not installed or model not found.
    """

    # Default voice — sounds natural and relatively deep
    DEFAULT_VOICE = "en_US-lessac-medium"

    def __init__(self, voice: str = DEFAULT_VOICE):
        self._voice_name = voice
        self._voice      = None
        self._lock       = threading.Lock()
        self._ready      = threading.Event()
        self._init_error = None
        # Load model in background so startup is non-blocking
        threading.Thread(target=self._load, daemon=True, name="piper-init").start()

    def _load(self):
        """Load the Piper voice model (downloads on first use if needed)."""
        try:
            from piper.voice import PiperVoice
            # Try to locate an ONNX model file in common locations
            model_paths = [
                Path.home() / ".local/share/piper" / f"{self._voice_name}.onnx",
                Path(__file__).parent / "voices" / f"{self._voice_name}.onnx",
                Path(f"{self._voice_name}.onnx"),
            ]
            model_file = next((p for p in model_paths if p.exists()), None)

            if model_file is None:
                log.info(f"[Piper] Model not found locally — downloading {self._voice_name}...")
                try:
                    from piper import download as piper_download
                    model_file = piper_download(self._voice_name)
                except Exception as dl_err:
                    raise RuntimeError(
                        f"Could not find or download Piper voice '{self._voice_name}'. "
                        f"Run: python -m piper --download-voices  ({dl_err})"
                    )

            self._voice = PiperVoice.load(str(model_file))
            log.info(f"[Piper] Voice loaded: {self._voice_name}")
            self._ready.set()

        except ImportError:
            self._init_error = (
                "piper-tts not installed. "
                "Run: pip install piper-tts  (falls back to pyttsx3)"
            )
            log.warning(f"[Piper] {self._init_error}")
            self._ready.set()
        except Exception as e:
            self._init_error = str(e)
            log.warning(f"[Piper] Init failed: {e}")
            self._ready.set()

    def _wait_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout=timeout) and self._voice is not None

    def speak(self, text: str):
        """Synthesise and play text synchronously."""
        if not text.strip():
            return
        if not self._wait_ready():
            log.warning("[Piper] Not ready — skipping utterance")
            return
        with self._lock:
            try:
                import sounddevice as sd
                import numpy as np

                cfg = self._voice.config
                sample_rate = cfg.sample_rate

                audio_chunks = []
                for raw in self._voice.synthesize_stream_raw(text):
                    audio_chunks.append(np.frombuffer(raw, dtype=np.int16))

                if audio_chunks:
                    audio = np.concatenate(audio_chunks).astype(np.float32) / 32768.0
                    sd.play(audio, sample_rate)
                    sd.wait()

            except Exception as e:
                log.error(f"[Piper] Speak error: {e}")

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def stop(self):
        pass

    @property
    def is_available(self) -> bool:
        return self._ready.is_set() and self._voice is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  PYTTSX3 TTS — fast pyttsx3 subprocess (original, always available)
# ═══════════════════════════════════════════════════════════════════════════════

class Pyttsx3TTS:
    """
    Windows TTS via subprocess isolation.
    Spawns a fresh Python subprocess per utterance to avoid COM thread issues.
    """

    _SAPI_SCRIPT = (
        "import sys, pyttsx3; "
        "e = pyttsx3.init(); "
        "e.setProperty('rate', 160); "
        "e.setProperty('volume', 1.0); "
        "e.say(sys.argv[1]); "
        "e.runAndWait()"
    )

    _PS_SCRIPT = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 1; $s.Volume = 100; "
        "$s.Speak([System.Text.RegularExpressions.Regex]::Replace($args[0], '[^\x20-\x7E]', ''));"
    )

    def __init__(self):
        self._init_error = None
        self._ready      = threading.Event()
        self._ready.set()
        self._use_ps = False
        try:
            import pyttsx3  # noqa
            log.info("[Voice] pyttsx3 TTS ready (subprocess mode)")
        except ImportError:
            self._use_ps = True
            log.warning("[Voice] pyttsx3 not found — using PowerShell SAPI fallback")

    def _run_subprocess(self, text: str) -> subprocess.Popen:
        safe = text.replace('"', "'").replace('\n', ' ').replace('\r', '')
        safe = ''.join(c for c in safe if ord(c) < 128)
        if self._use_ps:
            cmd = ["powershell", "-NonInteractive", "-Command", self._PS_SCRIPT, safe]
        else:
            cmd = [sys.executable, "-c", self._SAPI_SCRIPT, safe]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def speak(self, text: str):
        if not text:
            return
        proc = self._run_subprocess(text)
        proc.wait(timeout=30)

    def speak_async(self, text: str) -> threading.Thread:
        if not text:
            return threading.Thread()
        log.info(f"[Voice] TTS: {len(text)} chars")
        proc = self._run_subprocess(text)
        t = threading.Thread(target=proc.wait, daemon=True, name="tts-reaper")
        t.start()
        return t

    def stop(self):
        pass


class CoquiTTS:
    """Natural voice TTS using Coqui TTS — free, local, ~1.5GB model."""

    def __init__(self):
        self._model = None
        self._lock  = threading.Lock()

    def _init_model(self):
        if self._model is None:
            try:
                from TTS.api import TTS as CoquiTTSAPI
                log.info("[Voice] Loading Coqui TTS model...")
                self._model = CoquiTTSAPI(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                log.info("[Voice] Coqui TTS initialized")
            except ImportError:
                log.error("[Voice] Coqui TTS not installed. Run: pip install TTS")
                raise

    def speak(self, text: str):
        with self._lock:
            self._init_model()
            try:
                tmp_path = os.path.join(tempfile.gettempdir(), "jarvis_tts.wav")
                self._model.tts_to_file(text=text, file_path=tmp_path)
                self._play_audio(tmp_path)
            except Exception as e:
                log.error(f"[Voice] Coqui TTS error: {e}")

    def _play_audio(self, filepath: str):
        try:
            import sounddevice as sd
            import soundfile as sf
            data, samplerate = sf.read(filepath, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            try:
                subprocess.Popen(
                    f'start /min "" "{filepath}"', shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                log.error(f"[Voice] Could not play audio: {e}")

    def speak_async(self, text: str) -> threading.Thread:
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ═══════════════════════════════════════════════════════════════════════════════
#  STT ENGINE (faster-whisper) — unchanged, already fast
# ═══════════════════════════════════════════════════════════════════════════════

class WhisperSTT:
    """Speech-to-text using faster-whisper (tiny.en, ~39MB, int8)."""

    def __init__(self, model_size: str = "tiny.en"):
        self._model      = None
        self._model_size = model_size
        self._lock       = threading.Lock()

    def _init_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                log.info(f"[Voice] Loading Whisper STT ({self._model_size})...")
                self._model = WhisperModel(
                    self._model_size, device="cpu", compute_type="int8"
                )
                log.info("[Voice] Whisper STT initialized")
            except ImportError:
                log.error("[Voice] faster-whisper not installed. Run: pip install faster-whisper")
                raise

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        with self._lock:
            self._init_model()
            try:
                tmp_path = os.path.join(tempfile.gettempdir(), "jarvis_stt.wav")
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                segments, _ = self._model.transcribe(
                    tmp_path, beam_size=3, language="en", vad_filter=True,
                )
                text = " ".join(s.text for s in segments).strip()
                log.info(f"[Voice] Transcribed: {text[:80]}")
                return text
            except Exception as e:
                log.error(f"[Voice] Transcription error: {e}")
                return ""

    def transcribe_file(self, filepath: str) -> str:
        with self._lock:
            self._init_model()
            try:
                segments, _ = self._model.transcribe(
                    filepath, beam_size=3, language="en", vad_filter=True,
                )
                return " ".join(s.text for s in segments).strip()
            except Exception as e:
                log.error(f"[Voice] File transcription error: {e}")
                return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  VAD — tighter silence detection
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    """Detect speech using silero-vad."""

    def __init__(self, threshold: float = 0.5):
        self._model      = None
        self._threshold  = threshold
        self._unavailable = False

    def _init_model(self):
        if self._unavailable or self._model is not None:
            return
        try:
            import torch
            self._model, self._utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                verbose=False,
            )
            log.info("[Voice] Silero VAD initialized")
        except Exception as e:
            log.warning(f"[Voice] Silero VAD not available: {e}")
            self._unavailable = True

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        self._init_model()
        if self._model is None:
            return True
        try:
            import torch
            import numpy as np
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np)
            confidence = self._model(audio_tensor, sample_rate).item()
            return confidence > self._threshold
        except Exception:
            return True


# ═══════════════════════════════════════════════════════════════════════════════
#  WAKE WORD DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class WakeWordDetector:
    """Detect 'Hey JARVIS' using openwakeword."""

    def __init__(self):
        self._model = None

    def _init_model(self):
        if self._model is None:
            try:
                from openwakeword.model import Model
                self._model = Model(
                    wakeword_models=["hey_jarvis_v0.1"],
                    inference_framework="onnx",
                )
                log.info("[Voice] Wake word detector initialized")
            except ImportError:
                log.warning("[Voice] openwakeword not installed. Run: pip install openwakeword")
            except Exception as e:
                log.warning(f"[Voice] Wake word init failed: {e}")

    def detect(self, audio_chunk: bytes) -> bool:
        self._init_model()
        if self._model is None:
            return False
        try:
            import numpy as np
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            prediction = self._model.predict(audio_np)
            for key, score in prediction.items():
                if "hey_jarvis" in key and score > 0.3:
                    log.info(f"[Voice] Wake word detected! ({key}: {score:.3f})")
                    return True
            return False
        except Exception as e:
            log.debug(f"[Voice] Detect error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    """
    Complete voice system for JARVIS.

    NEW in this version:
        stream_speak(token_generator) — speaks sentence-by-sentence as tokens arrive
        Tighter silence detection: 0.5s default (was 1.5s)
        PiperTTS option for natural neural voice
    """

    def __init__(self, tts_engine: str = "pyttsx3"):
        self.is_speaking  = False
        self.is_listening = False
        self._stop_flag   = threading.Event()

        # ── TTS selection ────────────────────────────────────────────────
        if tts_engine == "piper":
            self._tts = PiperTTS()
            # If Piper fails to load, fall back to pyttsx3
            threading.Thread(target=self._check_piper_fallback, daemon=True).start()
        elif tts_engine == "coqui":
            self._tts = CoquiTTS()
        else:
            self._tts = Pyttsx3TTS()
        self._tts_engine_name = tts_engine

        self._stt       = None
        self._vad       = None
        self._wake_word = None

        log.info(f"[Voice] Engine initialized (TTS: {tts_engine})")

    def _check_piper_fallback(self):
        """After Piper's init window, fall back to pyttsx3 if unavailable."""
        time.sleep(12)
        if isinstance(self._tts, PiperTTS) and not self._tts.is_available:
            log.warning("[Voice] Piper unavailable — switching to pyttsx3 fallback")
            self._tts = Pyttsx3TTS()
            self._tts_engine_name = "pyttsx3_fallback"

    def _get_stt(self) -> WhisperSTT:
        if self._stt is None:
            self._stt = WhisperSTT("tiny.en")
        return self._stt

    def _get_vad(self) -> VoiceActivityDetector:
        if self._vad is None:
            self._vad = VoiceActivityDetector()
        return self._vad

    def _get_wake_word(self) -> WakeWordDetector:
        if self._wake_word is None:
            self._wake_word = WakeWordDetector()
        return self._wake_word

    # ── Standard TTS ─────────────────────────────────────────────────────

    def speak(self, text: str):
        if not text:
            return
        self.is_speaking = True
        try:
            self._tts.speak(text)
        except Exception as e:
            log.error(f"[Voice] TTS error: {e}")
        finally:
            self.is_speaking = False

    def speak_async(self, text: str) -> threading.Thread:
        self.is_speaking = True

        def _speak():
            try:
                self._tts.speak(text)
            except Exception as e:
                log.error(f"[Voice] TTS async error: {e}")
            finally:
                self.is_speaking = False

        t = threading.Thread(target=_speak, daemon=True)
        t.start()
        return t

    # ── STREAMING SPEAK — NEW ─────────────────────────────────────────────

    def stream_speak(self, token_generator, on_sentence: callable = None) -> StreamingSpeaker:
        """
        Real-time streaming TTS pipeline.

        Iterates a token generator (from think_stream), feeds each token into
        a StreamingSpeaker that starts playing audio as soon as the first
        sentence is complete — before the AI has even finished responding.

        Args:
            token_generator: Any iterable of string tokens.
            on_sentence: Optional callback(sentence_text) called before each utterance.

        Returns:
            The StreamingSpeaker (call .wait_done() to block until audio finishes).

        Example:
            gen = (chunk["token"] for chunk in think_stream(prompt) if "token" in chunk)
            speaker = engine.stream_speak(gen)
            speaker.wait_done()
        """

        class _InstrumentedSpeaker(StreamingSpeaker):
            def _enqueue(inner_self, sentence: str):
                if on_sentence and sentence.strip():
                    try:
                        on_sentence(sentence.strip())
                    except Exception:
                        pass
                super()._enqueue(sentence)

        speaker = _InstrumentedSpeaker(self._tts) if on_sentence else StreamingSpeaker(self._tts)

        def _run():
            try:
                for token in token_generator:
                    speaker.feed_token(token)
                speaker.flush()
            except Exception as e:
                log.error(f"[Voice] stream_speak error: {e}")

        threading.Thread(target=_run, daemon=True, name="stream-speak").start()
        return speaker

    def speak_streaming_text(self, text_generator):
        """
        Legacy method — speaks a text generator sentence-by-sentence.
        Kept for backward compat. Prefer stream_speak() for token streams.
        """
        buffer = ""
        for chunk in text_generator:
            buffer += chunk
            sentences = re.split(r'(?<=[.!?])\s+', buffer)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    if s.strip():
                        self.speak(s.strip())
                buffer = sentences[-1]
            if self._stop_flag.is_set():
                return
        if buffer.strip():
            self.speak(buffer.strip())

    def stop_speaking(self):
        self._stop_flag.set()
        self.is_speaking = False
        log.info("[Voice] Speech interrupted")

    # ── STT ─────────────────────────────────────────────────────────────

    def listen(
        self,
        timeout: float = 5.0,
        silence_timeout: float = 0.5,   # ← tightened from 1.5s to 0.5s
    ) -> str:
        """
        Listen for speech and transcribe it.

        silence_timeout is now 0.5s by default (was 1.5s) so the pipeline
        feels snappier — JARVIS stops waiting sooner after you finish speaking.
        """
        self.is_listening = True
        self._stop_flag.clear()

        try:
            import sounddevice as sd
            import numpy as np

            RATE     = 16000
            CHUNK    = 1024
            CHANNELS = 1

            frames        = []
            silence_start = None
            start_time    = time.time()
            vad           = self._get_vad()

            log.info("[Voice] Listening...")

            def callback(indata, frame_count, time_info, status):
                frames.append(indata.copy())

            with sd.InputStream(
                samplerate=RATE, channels=CHANNELS, dtype="int16",
                blocksize=CHUNK, callback=callback
            ):
                while not self._stop_flag.is_set():
                    if time.time() - start_time > timeout:
                        break
                    if frames:
                        chunk_bytes = frames[-1].tobytes()
                        has_speech  = vad.is_speech(chunk_bytes, RATE)
                        if has_speech:
                            silence_start = None
                        else:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > silence_timeout:
                                log.info("[Voice] Silence detected — stopping")
                                break
                    time.sleep(0.05)

            if not frames:
                self.is_listening = False
                return ""

            audio_data = np.concatenate(frames, axis=0).tobytes()
            stt        = self._get_stt()
            text       = stt.transcribe(audio_data, RATE)
            self.is_listening = False
            return text

        except ImportError:
            log.error("[Voice] sounddevice not installed. Run: pip install sounddevice")
            self.is_listening = False
            return ""
        except Exception as e:
            log.error(f"[Voice] Listen error: {e}")
            self.is_listening = False
            return ""

    # ── Wake Word ────────────────────────────────────────────────────────

    def listen_for_wake_word(self, callback=None):
        try:
            import sounddevice as sd
            import numpy as np
            RATE  = 16000
            CHUNK = 1280

            wake_detector = self._get_wake_word()
            log.info("[Voice] Listening for wake word ('Hey JARVIS')...")

            buf      = np.zeros((CHUNK,), dtype="int16")
            buf_lock = threading.Event()

            def _cb(indata, frames, time_info, status):
                nonlocal buf
                buf = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                buf_lock.set()

            with sd.InputStream(
                samplerate=RATE, channels=1, dtype="int16",
                blocksize=CHUNK, callback=_cb
            ):
                while not self._stop_flag.is_set():
                    buf_lock.wait(timeout=1.0)
                    buf_lock.clear()
                    if wake_detector.detect(buf.tobytes()):
                        if callback:
                            callback()
                        return True
            return False

        except ImportError:
            log.error("[Voice] sounddevice not installed.")
            return False
        except Exception as e:
            log.error(f"[Voice] Wake word error: {e}")
            return False

    # ── Engine management ────────────────────────────────────────────────

    def switch_tts(self, engine: str):
        """Switch TTS engine at runtime: 'pyttsx3', 'piper', or 'coqui'."""
        if engine == "piper":
            self._tts = PiperTTS()
        elif engine == "coqui":
            self._tts = CoquiTTS()
        else:
            self._tts = Pyttsx3TTS()
        self._tts_engine_name = engine
        log.info(f"[Voice] TTS switched to: {engine}")

    def get_status(self) -> dict:
        return {
            "tts_engine":       self._tts_engine_name,
            "is_speaking":      self.is_speaking,
            "is_listening":     self.is_listening,
            "stt_loaded":       self._stt is not None,
            "vad_loaded":       self._vad is not None,
            "wake_word_loaded": self._wake_word is not None,
            "piper_available":  isinstance(self._tts, PiperTTS) and self._tts.is_available,
        }

    def stop(self):
        self._stop_flag.set()
        self.is_speaking  = False
        self.is_listening = False


# ═══════════════════════════════════════════════════════════════════════════════
#  WAKE WORD SERVICE (unchanged from original, already solid)
# ═══════════════════════════════════════════════════════════════════════════════

class WakeWordService:
    """
    Background service: listen for 'Hey JARVIS', capture command, publish transcript.

    Events pushed to subscribers:
        {"event": "detected"}
        {"event": "listening"}
        {"event": "transcript", "text": "..."}
        {"event": "error", "message": "..."}
        {"event": "heartbeat"}
    """

    def __init__(self):
        self._enabled     = False
        self._thread      = None
        self._stop        = threading.Event()
        self._subscribers: list[queue.Queue] = []
        self._sub_lock    = threading.Lock()
        self._voice: VoiceEngine | None = None

    def subscribe(self):
        q: queue.Queue = queue.Queue(maxsize=32)
        with self._sub_lock:
            self._subscribers.append(q)
        try:
            while True:
                try:
                    yield q.get(timeout=30)
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

    def enable(self):
        if self._enabled:
            return
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="WakeWordService"
        )
        self._thread.start()
        log.info("[WakeWord] Service enabled")

    def disable(self):
        self._enabled = False
        self._stop.set()
        log.info("[WakeWord] Service disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def _loop(self):
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as e:
            self._publish({"event": "unavailable", "message": f"Missing: {e}"})
            return

        if self._voice is None:
            self._voice = get_voice_engine()

        RATE  = 16000
        CHUNK = 1280

        log.info("[WakeWord] Listening for 'Hey JARVIS'...")

        while not self._stop.is_set():
            try:
                detector   = self._voice._get_wake_word()
                buf_ready  = threading.Event()
                latest_buf = [None]

                def callback(indata, frames, time_info, status):
                    latest_buf[0] = (
                        indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                    )
                    buf_ready.set()

                with sd.InputStream(
                    samplerate=RATE, channels=1, dtype="int16",
                    blocksize=CHUNK, callback=callback
                ):
                    while not self._stop.is_set():
                        buf_ready.wait(timeout=1.0)
                        buf_ready.clear()
                        if latest_buf[0] is not None:
                            if detector.detect(latest_buf[0].tobytes()):
                                log.info("[WakeWord] Detected!")
                                self._publish({"event": "detected"})
                                break

                if self._stop.is_set():
                    break

                self._publish({"event": "listening"})
                time.sleep(0.15)

                # Use tighter silence timeout for wake-word flow
                transcript = self._voice.listen(timeout=6.0, silence_timeout=0.5)

                if transcript and transcript.strip():
                    log.info(f"[WakeWord] Transcript: {transcript}")
                    self._publish({"event": "transcript", "text": transcript.strip()})
                else:
                    self._publish({"event": "error", "message": "Didn't catch that. Try again."})

                time.sleep(0.5)

            except Exception as e:
                log.error(f"[WakeWord] Loop error: {e}")
                self._publish({"event": "error", "message": str(e)})
                time.sleep(2.0)

        log.info("[WakeWord] Service stopped")


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_voice_instance    = None
_voice_lock        = threading.Lock()
_wake_service      = None
_wake_service_lock = threading.Lock()


def get_voice_engine(tts_engine: str = "pyttsx3") -> VoiceEngine:
    """Get the global VoiceEngine singleton."""
    global _voice_instance
    if _voice_instance is None:
        with _voice_lock:
            if _voice_instance is None:
                # Auto-try Piper; fall back to pyttsx3 if not installed
                try:
                    import piper  # noqa
                    tts = "piper"
                except ImportError:
                    tts = tts_engine
                _voice_instance = VoiceEngine(tts)
    return _voice_instance


def get_wake_service() -> WakeWordService:
    global _wake_service
    if _wake_service is None:
        with _wake_service_lock:
            if _wake_service is None:
                _wake_service = WakeWordService()
    return _wake_service


# ── Backward-compat helpers ──────────────────────────────────────────────────

def listen() -> str:
    try:
        return get_voice_engine().listen()
    except Exception:
        return ""

def speak(text: str) -> None:
    try:
        get_voice_engine().speak_async(text)
    except Exception:
        pass
"""
voice.py — JARVIS Voice Engine (100% Free, Local)
====================================================
Modular voice system with STT, TTS, VAD, and wake word detection.
All components are free and run locally — no API keys needed.

Stack:
    STT:       faster-whisper (tiny.en, ~39MB, <100ms)
    TTS:       pyttsx3 (instant, offline) + Coqui TTS (natural voice)
    VAD:       silero-vad (voice activity detection)
    Wake Word: openwakeword — custom ONNX model loaded via model_paths
    Audio:     sounddevice (mic input)

Wake word model path (custom-trained):
    wake/models/hey_jarvis_v0.1.onnx

Usage:
    from voice import VoiceEngine
    engine = VoiceEngine()
    engine.speak("Good morning, sir.")
    text = engine.listen()
"""

import os
import io
import sys
import wave
import time
import logging
import subprocess
import threading
import tempfile
from pathlib import Path

log = logging.getLogger("jarvis.voice")

# ── Resolve the custom ONNX model path relative to this file ────────────────
# Layout expected:
#   jarvis/
#     voice.py          ← this file
#     wake/
#       models/
#         hey_jarvis_v0.1.onnx   ← custom-trained model
#
# Override by setting env var JARVIS_WAKE_MODEL to an absolute path.
_VOICE_DIR = Path(__file__).parent
_DEFAULT_WAKE_MODEL = _VOICE_DIR / "wake" / "models" / "hey_jarvis_custom.onnx"
WAKE_MODEL_PATH = Path(os.environ.get("JARVIS_WAKE_MODEL", str(_DEFAULT_WAKE_MODEL)))


# ═══════════════════════════════════════════════════════════════════════════════
#  TTS ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

class Pyttsx3TTS:
    """
    Windows TTS via subprocess isolation.

    pyttsx3 with SAPI5 (Windows COM) has strict thread-affinity requirements.
    Even with a dedicated worker thread, runAndWait() can silently drop audio
    when called from a non-main Flask thread because the COM STA message pump
    doesn't always get properly serviced.

    Fix: spawn a fresh Python subprocess per utterance.  The subprocess owns
    its own main thread, its own COM context, and its own SAPI5 session.
    Completely bypasses all threading issues.  speak_async() is fire-and-forget;
    speak() blocks until the subprocess exits.
    """

    # One-liner script run in each subprocess
    _SAPI_SCRIPT = (
        "import sys, pyttsx3; "
        "e = pyttsx3.init(); "
        "e.setProperty('rate', 160); "
        "e.setProperty('volume', 1.0); "
        "e.say(sys.argv[1]); "
        "e.runAndWait()"
    )

    # PowerShell fallback (always available on Windows, no pip deps)
    _PS_SCRIPT = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = 1; $s.Volume = 100; "
        "$s.Speak([System.Text.RegularExpressions.Regex]::Replace($args[0], '[^\x20-\x7E]', ''));"
    )

    def __init__(self):
        self._init_error = None
        self._ready = threading.Event()
        self._ready.set()    # always ready — no warmup needed
        # Detect preferred method
        self._use_ps = False
        try:
            import pyttsx3  # noqa: just check importability
            log.info("[Voice] pyttsx3 TTS ready (subprocess mode)")
        except ImportError:
            self._use_ps = True
            log.warning("[Voice] pyttsx3 not found — falling back to PowerShell SAPI")

    def _run_subprocess(self, text: str) -> subprocess.Popen:
        """Launch a TTS subprocess and return the Popen handle."""
        # Sanitise: remove characters that would break the script
        safe = text.replace('"', "'").replace('\n', ' ').replace('\r', '')
        safe = ''.join(c for c in safe if ord(c) < 128)   # ASCII only for SAPI5

        if self._use_ps:
            cmd = ["powershell", "-NonInteractive", "-Command",
                   self._PS_SCRIPT, safe]
        else:
            cmd = [sys.executable, "-c", self._SAPI_SCRIPT, safe]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def speak(self, text: str):
        """Speak text and block until playback finishes."""
        if not text:
            return
        proc = self._run_subprocess(text)
        proc.wait(timeout=30)

    def speak_async(self, text: str):
        """Spawn TTS subprocess and return immediately (fire-and-forget)."""
        if not text:
            return threading.Thread()
        log.info(f"[Voice] TTS speaking ({len(text)} chars)")
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
        self._lock = threading.Lock()

    def _init_model(self):
        if self._model is None:
            try:
                from TTS.api import TTS as CoquiTTSAPI
                log.info("[Voice] Loading Coqui TTS model (first run downloads ~1.5GB)...")
                self._model = CoquiTTSAPI(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                log.info("[Voice] Coqui TTS initialized")
            except ImportError:
                log.error("[Voice] Coqui TTS not installed! Run: pip install TTS")
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

    def speak_async(self, text: str):
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t


# ═══════════════════════════════════════════════════════════════════════════════
#  STT ENGINE (faster-whisper)
# ═══════════════════════════════════════════════════════════════════════════════

class WhisperSTT:
    """Speech-to-text using faster-whisper (tiny.en, ~39MB, free)."""

    def __init__(self, model_size: str = "tiny.en"):
        self._model = None
        self._model_size = model_size
        self._lock = threading.Lock()

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
                log.error("[Voice] faster-whisper not installed! Run: pip install faster-whisper")
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
                    tmp_path, beam_size=3, language="en",
                    vad_filter=True,
                )
                text = " ".join(s.text for s in segments).strip()
                log.info(f"[Voice] Transcribed: {text[:80]}...")
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
#  VOICE ACTIVITY DETECTION (silero-vad)
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceActivityDetector:
    """Detect when the user starts/stops speaking using silero-vad."""

    def __init__(self, threshold: float = 0.5):
        self._model = None
        self._threshold = threshold
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
            log.warning(f"[Voice] Silero VAD not available: {e} — will assume speech for all chunks")
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
#  WAKE WORD DETECTION (openwakeword — custom ONNX via model_paths)
# ═══════════════════════════════════════════════════════════════════════════════

class WakeWordDetector:
    """
    Detect "Hey JARVIS" using a custom-trained ONNX model.

    KEY DESIGN DECISION — model_paths vs wakeword_models
    ─────────────────────────────────────────────────────
    openwakeword's `wakeword_models` parameter accepts *registered* built-in
    model names (e.g. "alexa", "hey_mycroft").  Passing an unrecognised string
    there causes it to silently fall back to whatever is cached — which can
    include Alexa or other default models, causing false triggers on the wrong
    wake phrase.

    `model_paths` accepts a list of *absolute or relative filesystem paths*
    to .onnx files and is the correct parameter for custom-trained models.
    Using it guarantees only the specified model is loaded; no built-in or
    cached models are pulled in alongside it.
    """

    DETECTION_THRESHOLD = 0.5   # raise toward 0.7 to reduce false positives

    def __init__(self):
        self._model = None
        self._model_path = WAKE_MODEL_PATH

    def _init_model(self):
        if self._model is not None:
            return

        # ── Guard: verify the ONNX file actually exists ─────────────────
        if not self._model_path.exists():
            log.error(
                f"[WakeWord] Custom model NOT FOUND at: {self._model_path}\n"
                f"           Train it with train_wake.py, then place the output at:\n"
                f"           {self._model_path}\n"
                f"           Or set env var JARVIS_WAKE_MODEL to the correct path."
            )
            return

        try:
            from openwakeword.model import Model

            log.info("─" * 60)
            log.info("[WakeWord] Initialising custom wake-word engine")
            log.info(f"[WakeWord]   Model path  : {self._model_path}")
            log.info(f"[WakeWord]   Wake phrase : Hey JARVIS")
            log.info(f"[WakeWord]   Threshold   : {self.DETECTION_THRESHOLD}")

            # ── CRITICAL: use model_paths, NOT wakeword_models ───────────
            # model_paths loads the file directly from disk.
            # wakeword_models looks up a registry of built-in names and
            # would fall back to cached/default models (including Alexa)
            # when the name is not recognised.
            self._model = Model(
                wakeword_models=[],          # explicitly empty — no built-ins
                model_paths=[str(self._model_path)],   # custom ONNX only
                inference_framework="onnx",
            )

            # Log the model names that openwakeword actually loaded so we
            # can confirm there are no unwanted built-in models present.
            loaded_names = list(self._model.models.keys()) if hasattr(self._model, "models") else ["(unavailable)"]
            log.info(f"[WakeWord]   Loaded keys : {loaded_names}")
            log.info("[WakeWord] ✓ Wake word detector ready — listening for 'Hey JARVIS'")
            log.info("─" * 60)

        except ImportError:
            log.warning(
                "[WakeWord] openwakeword not installed. "
                "Run: pip install openwakeword"
            )
        except Exception as e:
            log.error(f"[WakeWord] Initialisation failed: {e}")

    def detect(self, audio_chunk: bytes) -> bool:
        """
        Return True if the audio chunk contains the 'Hey JARVIS' wake phrase.

        Only scores from the custom model are examined.  No built-in model
        scores are checked so Alexa / other phrases cannot trigger detection.
        """
        self._init_model()
        if self._model is None:
            return False
        try:
            import numpy as np
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            prediction = self._model.predict(audio_np)

            for key, score in prediction.items():
                # Match only keys that relate to our custom hey_jarvis model.
                # Reject anything that looks like a built-in (alexa, mycroft, etc.)
                key_lower = key.lower()
                if "hey_jarvis" in key_lower or "jarvis" in key_lower:
                    if score > self.DETECTION_THRESHOLD:
                        log.info(f"[WakeWord] ✓ DETECTED — model='{key}' score={score:.3f}")
                        return True
                else:
                    # Log unexpected keys at debug level so you can spot
                    # any rogue built-in models that shouldn't be there.
                    log.debug(f"[WakeWord] Ignoring non-JARVIS model key: '{key}' score={score:.3f}")
            return False

        except Exception as e:
            log.debug(f"[WakeWord] Detect error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    """
    Complete voice system for JARVIS.
    All components are free and run locally.
    """

    def __init__(self, tts_engine: str = "pyttsx3"):
        self.is_speaking = False
        self.is_listening = False
        self._stop_flag = threading.Event()

        if tts_engine == "coqui":
            self._tts = CoquiTTS()
        else:
            self._tts = Pyttsx3TTS()
        self._tts_engine_name = tts_engine

        self._stt = None
        self._vad = None
        self._wake_word = None
        self._pyaudio = None

        log.info(f"[Voice] Engine initialized (TTS: {tts_engine})")

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

    # ── TTS ─────────────────────────────────────────────────────────────

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
                log.error(f"[Voice] TTS error: {e}")
            finally:
                self.is_speaking = False

        t = threading.Thread(target=_speak, daemon=True)
        t.start()
        return t

    def speak_streaming(self, text_generator):
        buffer = ""
        sentence_ends = ".!?;\n"

        for chunk in text_generator:
            buffer += chunk
            for i, char in enumerate(buffer):
                if char in sentence_ends and i > 10:
                    sentence = buffer[:i+1].strip()
                    buffer = buffer[i+1:]
                    if sentence:
                        self.speak(sentence)
                    if self._stop_flag.is_set():
                        return
                    break

        if buffer.strip():
            self.speak(buffer.strip())

    def stop_speaking(self):
        self._stop_flag.set()
        self.is_speaking = False
        log.info("[Voice] Speech interrupted")

    # ── STT ─────────────────────────────────────────────────────────────

    def listen(self, timeout: float = 5.0, silence_timeout: float = 1.5) -> str:
        self.is_listening = True
        self._stop_flag.clear()

        try:
            import sounddevice as sd
            import numpy as np

            RATE = 16000
            CHUNK = 1024
            CHANNELS = 1

            frames = []
            silence_start = None
            start_time = time.time()
            vad = self._get_vad()

            log.info("[Voice] Listening...")

            def callback(indata, frame_count, time_info, status):
                frames.append(indata.copy())

            with sd.InputStream(samplerate=RATE, channels=CHANNELS,
                                dtype="int16", blocksize=CHUNK,
                                callback=callback):
                while not self._stop_flag.is_set():
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        break
                    if frames:
                        chunk_bytes = frames[-1].tobytes()
                        has_speech = vad.is_speech(chunk_bytes, RATE)
                        if has_speech:
                            silence_start = None
                        else:
                            if silence_start is None:
                                silence_start = time.time()
                            elif time.time() - silence_start > silence_timeout:
                                log.info("[Voice] Silence detected, stopping...")
                                break
                    time.sleep(0.05)

            if not frames:
                self.is_listening = False
                return ""

            audio_data = np.concatenate(frames, axis=0).tobytes()
            stt = self._get_stt()
            text = stt.transcribe(audio_data, RATE)
            self.is_listening = False
            return text

        except ImportError:
            log.error("[Voice] sounddevice not installed! Run: pip install sounddevice")
            self.is_listening = False
            return ""
        except Exception as e:
            log.error(f"[Voice] Listen error: {e}")
            self.is_listening = False
            return ""

    # ── Wake Word ───────────────────────────────────────────────────────

    def listen_for_wake_word(self, callback=None):
        try:
            import sounddevice as sd
            import numpy as np
            RATE = 16000
            CHUNK = 1280

            wake_detector = self._get_wake_word()
            log.info("[Voice] Listening for wake word ('Hey JARVIS')...")

            buf = np.zeros((CHUNK,), dtype="int16")
            buf_lock = threading.Event()

            def audio_callback(indata, frames, time_info, status):
                nonlocal buf
                buf = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                buf_lock.set()

            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                blocksize=CHUNK, callback=audio_callback):
                while not self._stop_flag.is_set():
                    buf_lock.wait(timeout=1.0)
                    buf_lock.clear()
                    if wake_detector.detect(buf.tobytes()):
                        if callback:
                            callback()
                        return True

            return False

        except ImportError:
            log.error("[Voice] sounddevice not installed! Run: pip install sounddevice")
            return False
        except Exception as e:
            log.error(f"[Voice] Wake word error: {e}")
            return False

    # ── Engine Management ───────────────────────────────────────────────

    def switch_tts(self, engine: str):
        if engine == "coqui":
            self._tts = CoquiTTS()
        else:
            self._tts = Pyttsx3TTS()
        self._tts_engine_name = engine
        log.info(f"[Voice] TTS switched to: {engine}")

    def get_status(self) -> dict:
        return {
            "tts_engine": self._tts_engine_name,
            "is_speaking": self.is_speaking,
            "is_listening": self.is_listening,
            "stt_loaded": self._stt is not None,
            "vad_loaded": self._vad is not None,
            "wake_word_loaded": self._wake_word is not None,
            "wake_model_path": str(WAKE_MODEL_PATH),
            "wake_model_exists": WAKE_MODEL_PATH.exists(),
        }

    def stop(self):
        self._stop_flag.set()
        self.is_speaking = False
        self.is_listening = False


# ═══════════════════════════════════════════════════════════════════════════════
#  WAKE WORD SERVICE — background daemon with SSE event queue
# ═══════════════════════════════════════════════════════════════════════════════

import queue

class WakeWordService:
    """
    Background service that continuously listens for "Hey JARVIS",
    then immediately captures and transcribes the follow-up command.
    """

    def __init__(self):
        self._enabled = False
        self._thread = None
        self._stop = threading.Event()
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
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
        self._thread = threading.Thread(target=self._loop, daemon=True, name="WakeWordService")
        self._thread.start()
        log.info("[WakeWord] Service enabled")

    def disable(self):
        self._enabled = False
        self._stop.set()
        log.info("[WakeWord] Service disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    def _loop(self):
        """Continuously detect wake word → capture command → publish transcript."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as e:
            self._publish({
                "event": "unavailable",
                "message": f"Missing dependency: {e}. Run: pip install sounddevice numpy"
            })
            log.error(f"[WakeWord] Missing dep: {e}")
            return

        if self._voice is None:
            self._voice = get_voice_engine()

        # Pre-initialise the detector now so the model path error (if any)
        # surfaces immediately rather than silently on first detection attempt.
        detector = self._voice._get_wake_word()
        detector._init_model()

        if detector._model is None:
            self._publish({
                "event": "unavailable",
                "message": (
                    f"Wake model not found at {WAKE_MODEL_PATH}. "
                    "Train it with train_wake.py and place the .onnx file there."
                )
            })
            return

        RATE = 16000
        CHUNK = 1280   # 80ms at 16kHz — required by openwakeword

        log.info("[WakeWord] Listening for 'Hey JARVIS'...")

        while not self._stop.is_set():
            try:
                buf_ready = threading.Event()
                latest_buf = [None]

                def callback(indata, frames, time_info, status):
                    latest_buf[0] = (indata[:, 0].copy() if indata.ndim > 1 else indata.copy())
                    buf_ready.set()

                # ── Phase 1: wait for wake word ──────────────────────────
                with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                    blocksize=CHUNK, callback=callback):
                    while not self._stop.is_set():
                        buf_ready.wait(timeout=1.0)
                        buf_ready.clear()
                        if latest_buf[0] is not None:
                            if detector.detect(latest_buf[0].tobytes()):
                                log.info("[WakeWord] 'Hey JARVIS' detected!")
                                self._publish({"event": "detected"})
                                break

                if self._stop.is_set():
                    break

                # ── Phase 2: capture follow-up command via STT ───────────
                self._publish({"event": "listening"})
                log.info("[WakeWord] Capturing command...")

                time.sleep(0.15)  # give hardware 150ms to release before reopening

                transcript = self._voice.listen(timeout=6.0, silence_timeout=1.2)

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
#  SINGLETON INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

_voice_instance = None
_voice_lock = threading.Lock()

def get_voice_engine(tts_engine: str = "pyttsx3") -> VoiceEngine:
    global _voice_instance
    if _voice_instance is None:
        with _voice_lock:
            if _voice_instance is None:
                _voice_instance = VoiceEngine(tts_engine)
    return _voice_instance


# ── Backward-compatible interface ────────────────────────────────────────────

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


# ── Wake Word Service singleton ──────────────────────────────────────────────

_wake_service_instance = None
_wake_service_lock = threading.Lock()

def get_wake_service() -> "WakeWordService":
    global _wake_service_instance
    if _wake_service_instance is None:
        with _wake_service_lock:
            if _wake_service_instance is None:
                _wake_service_instance = WakeWordService()
    return _wake_service_instance
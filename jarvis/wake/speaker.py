"""
wake/speaker.py — Speaker Verification
========================================
Uses Resemblyzer to build a d-vector embedding of the owner's voice
and verify that incoming audio matches before accepting a wake word.

This prevents JARVIS from responding to a TV, visitor, or recording.

Requirements:
    pip install resemblyzer

Usage:
    # Enroll (run once)
    sv = SpeakerVerifier()
    sv.enroll_from_mic(n_samples=5)

    # Verify
    audio_np = ...  # int16 numpy array
    ok, score = sv.verify(audio_np)
    if ok:
        print("Owner confirmed")
"""

import logging
import threading
import time
import json
from pathlib import Path

import numpy as np

from .config import (
    SAMPLE_RATE, SPEAKER_VERIFY_ENABLED,
    SPEAKER_SIMILARITY_MIN, SPEAKER_ENROLL_SECONDS,
    SPEAKER_ENROLL_COUNT, EMBEDDINGS_DIR,
)

log = logging.getLogger("jarvis.wake.speaker")

EMBEDDING_FILE = EMBEDDINGS_DIR / "owner_embedding.npy"
THRESHOLD_FILE = EMBEDDINGS_DIR / "threshold.json"


class SpeakerVerifier:
    """
    Owner voice verification using Resemblyzer d-vectors.

    Enrollment builds an average embedding from N speech samples.
    Verification computes cosine similarity between incoming audio
    and the stored embedding. Similarity >= threshold → accept.
    """

    def __init__(self):
        self._encoder = None
        self._owner_embedding = None
        self._threshold = SPEAKER_SIMILARITY_MIN
        self._lock = threading.Lock()
        self._available = False
        self._init()
        self._load_embedding()

    def _init(self):
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            self._available = True
            log.info("[Speaker] Resemblyzer VoiceEncoder ready")
        except ImportError:
            log.warning(
                "[Speaker] resemblyzer not installed — speaker verification disabled. "
                "Run: pip install resemblyzer"
            )
        except Exception as e:
            log.warning(f"[Speaker] Init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def enrolled(self) -> bool:
        return self._owner_embedding is not None

    # ── Enrollment ───────────────────────────────────────────────────────────

    def enroll_from_mic(self, n_samples: int = SPEAKER_ENROLL_COUNT,
                        seconds_each: float = SPEAKER_ENROLL_SECONDS) -> bool:
        """
        Record N samples from the mic and build the owner embedding.
        Prints prompts to stdout — call from a dedicated enrollment script.

        Args:
            n_samples: Number of speech samples to record.
            seconds_each: Duration of each sample in seconds.

        Returns:
            True if enrollment succeeded.
        """
        if not self._available:
            log.error("[Speaker] Resemblyzer not available — cannot enroll")
            return False

        try:
            import sounddevice as sd
        except ImportError:
            log.error("[Speaker] sounddevice not installed — cannot record")
            return False

        print("\n[JARVIS Speaker Enrollment]")
        print(f"Recording {n_samples} samples × {seconds_each}s each.")
        print("Speak naturally — count to ten, read a sentence, etc.\n")

        embeddings = []
        for i in range(n_samples):
            input(f"  Sample {i+1}/{n_samples} — press Enter then speak...")
            audio_np = self._record_raw(seconds_each, sd)
            emb = self._embed(audio_np)
            if emb is not None:
                embeddings.append(emb)
                print(f"  ✓ Captured (embedding dim {len(emb)})")
            else:
                print("  ✗ Failed to embed — skipping")

        if not embeddings:
            log.error("[Speaker] No valid embeddings captured")
            return False

        owner_emb = np.mean(embeddings, axis=0)
        self._save_embedding(owner_emb)
        self._owner_embedding = owner_emb
        print(f"\n  Enrolled! Embedding saved → {EMBEDDING_FILE}")
        print(f"  Using similarity threshold: {self._threshold:.2f}\n")
        return True

    def enroll_from_files(self, wav_paths: list) -> bool:
        """
        Build owner embedding from a list of WAV file paths.

        Args:
            wav_paths: List of str/Path pointing to 16kHz mono WAV files.

        Returns:
            True if enrollment succeeded.
        """
        if not self._available:
            return False

        embeddings = []
        for path in wav_paths:
            try:
                import soundfile as sf
                audio, sr = sf.read(str(path))
                if sr != SAMPLE_RATE:
                    log.warning(f"[Speaker] {path} is {sr}Hz — expected {SAMPLE_RATE}Hz")
                audio_np = (audio * 32767).astype(np.int16) if audio.dtype == np.float32 else audio
                emb = self._embed(audio_np)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                log.warning(f"[Speaker] Failed to load {path}: {e}")

        if not embeddings:
            return False

        owner_emb = np.mean(embeddings, axis=0)
        self._save_embedding(owner_emb)
        self._owner_embedding = owner_emb
        log.info(f"[Speaker] Enrolled from {len(embeddings)} files")
        return True

    # ── Verification ─────────────────────────────────────────────────────────

    def verify(self, audio_np: np.ndarray) -> tuple[bool, float]:
        """
        Verify whether audio matches the enrolled owner voice.

        Args:
            audio_np: int16 numpy array of audio.

        Returns:
            (accepted: bool, similarity_score: float 0–1)
        """
        if not self._available or not self.enrolled:
            # If not set up, pass all through
            return True, 1.0

        emb = self._embed(audio_np)
        if emb is None:
            return True, 1.0   # fail open — don't block on embedding errors

        score = float(np.dot(emb, self._owner_embedding) /
                      (np.linalg.norm(emb) * np.linalg.norm(self._owner_embedding) + 1e-8))
        accepted = score >= self._threshold
        log.debug(f"[Speaker] Similarity: {score:.3f} (threshold {self._threshold:.2f}) → {'✓' if accepted else '✗'}")
        return accepted, score

    def set_threshold(self, threshold: float):
        """Adjust the acceptance threshold at runtime."""
        self._threshold = max(0.0, min(1.0, threshold))
        self._save_threshold()
        log.info(f"[Speaker] Threshold updated to {self._threshold:.2f}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _embed(self, audio_np: np.ndarray) -> np.ndarray | None:
        """Generate a d-vector embedding for the given audio."""
        if not self._available:
            return None
        try:
            from resemblyzer import preprocess_wav
            # Resemblyzer needs float32 normalized to [-1, 1]
            audio_f = audio_np.astype(np.float32) / 32768.0
            # preprocess_wav handles resampling / trimming
            wav = preprocess_wav(audio_f, source_sr=SAMPLE_RATE)
            with self._lock:
                emb = self._encoder.embed_utterance(wav)
            return emb
        except Exception as e:
            log.debug(f"[Speaker] Embed error: {e}")
            return None

    def _record_raw(self, seconds: float, sd) -> np.ndarray:
        """Record audio from the default mic. Returns int16 array."""
        samples = int(SAMPLE_RATE * seconds)
        recording = sd.rec(samples, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        return recording.flatten()

    def _save_embedding(self, emb: np.ndarray):
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(EMBEDDING_FILE), emb)
        self._save_threshold()

    def _save_threshold(self):
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(THRESHOLD_FILE, "w") as f:
            json.dump({"threshold": self._threshold}, f)

    def _load_embedding(self):
        if EMBEDDING_FILE.exists():
            try:
                self._owner_embedding = np.load(str(EMBEDDING_FILE))
                log.info(f"[Speaker] Loaded owner embedding from {EMBEDDING_FILE}")
            except Exception as e:
                log.warning(f"[Speaker] Could not load embedding: {e}")

        if THRESHOLD_FILE.exists():
            try:
                with open(THRESHOLD_FILE) as f:
                    data = json.load(f)
                self._threshold = data.get("threshold", SPEAKER_SIMILARITY_MIN)
            except Exception:
                pass

    def get_status(self) -> dict:
        return {
            "available": self._available,
            "enrolled": self.enrolled,
            "threshold": self._threshold,
            "embedding_file": str(EMBEDDING_FILE) if self.enrolled else None,
        }

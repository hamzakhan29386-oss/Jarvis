"""
wake/collect_dataset.py — Wake Word Dataset Collection Tool
============================================================
Records "Hey Jarvis" samples with quality validation.
Target: 200–1000 positive clips, 500+ negative clips.

Usage:
    python -m wake.collect_dataset positive   # record wake word clips
    python -m wake.collect_dataset negative   # record background noise
    python -m wake.collect_dataset validate   # check dataset quality
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATASET  = BASE_DIR / "dataset"
POS_DIR  = DATASET / "positive"
NEG_DIR  = DATASET / "negative"
AUG_DIR  = DATASET / "augmented"
VAL_DIR  = DATASET / "validation"

SAMPLE_RATE    = 16_000
CHANNELS       = 1
SAMPWIDTH      = 2           # int16
RECORD_SECS    = 2.0         # seconds per clip
MIN_RMS        = 300         # reject silent clips
MAX_CLIPS_WARN = 1000        # warn if exceeded


def _make_dirs():
    for d in (POS_DIR, NEG_DIR, AUG_DIR, VAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _count(directory: Path) -> int:
    return len(list(directory.glob("*.wav")))


def _next_path(directory: Path, prefix: str) -> Path:
    idx = _count(directory) + 1
    return directory / f"{prefix}_{idx:04d}.wav"


def save_wav(path: Path, data: np.ndarray):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())


def record_clip(duration: float = RECORD_SECS) -> np.ndarray:
    """Record a single clip from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("  ERROR: sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    frames = []

    def cb(indata, f, t, status):
        frames.append(indata.copy())

    n_samples = int(SAMPLE_RATE * duration)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="int16", blocksize=1024, callback=cb):
        time.sleep(duration)

    audio = np.concatenate(frames, axis=0).flatten()
    return audio[:n_samples] if len(audio) >= n_samples else audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def validate_clip(audio: np.ndarray, label: str) -> tuple[bool, str]:
    """Quality checks for a recorded clip."""
    r = rms(audio)
    if r < MIN_RMS:
        return False, f"Too quiet (RMS={r:.0f} < {MIN_RMS})"
    if len(audio) < SAMPLE_RATE * 0.5:
        return False, "Too short"
    return True, f"OK (RMS={r:.0f})"


# ─── COLLECTION SESSIONS ─────────────────────────────────────────────────────

POSITIVE_PROMPTS = [
    "Normal tone",
    "Slightly louder",
    "Slightly quieter",
    "Faster pace",
    "Slower pace",
    "From 1 meter away",
    "From 3 meters away",
    "With TV noise in background",
    "With fan/AC running",
    "In a different room",
    "Sleepy/tired voice",
    "Excited/urgent tone",
    "Whispering",
    "After clearing throat",
    "With emphasis on HEY",
    "With emphasis on JARVIS",
    "Natural conversational",
    "Commanding tone",
    "Slightly distorted (cover mouth a bit)",
    "With keyboard sounds in background",
]


def session_positive(n_clips: int = 20, start_at: int = 0):
    """Interactive positive sample recording session."""
    _make_dirs()
    existing = _count(POS_DIR)
    print(f"\n{'='*60}")
    print("  POSITIVE SAMPLE COLLECTION — Say 'Hey JARVIS'")
    print(f"  Existing clips: {existing}")
    print(f"  Recording {n_clips} new clips")
    print(f"  Duration per clip: {RECORD_SECS}s")
    print("  Tip: vary tone, distance, speed, environment.")
    print(f"{'='*60}\n")

    input("  Press ENTER to begin...")

    good = 0
    bad  = 0

    for i in range(n_clips):
        prompt_idx = (start_at + i) % len(POSITIVE_PROMPTS)
        prompt = POSITIVE_PROMPTS[prompt_idx]

        print(f"\n  [{i+1}/{n_clips}] Style: {prompt}")
        print("  Get ready...", end="", flush=True)
        time.sleep(0.8)
        print(" → SPEAK NOW ←", end="", flush=True)

        audio = record_clip()

        ok, msg = validate_clip(audio, "positive")
        if ok:
            path = _next_path(POS_DIR, "hey_jarvis")
            save_wav(path, audio)
            good += 1
            print(f" ✓ Saved ({msg})")
        else:
            bad += 1
            print(f" ✗ Rejected: {msg} — skipped")

        time.sleep(0.4)

    print(f"\n  Session done. Good: {good}  Rejected: {bad}")
    print(f"  Total positive clips: {_count(POS_DIR)}")


def session_negative(n_clips: int = 30):
    """Passive negative sample recording (background audio)."""
    _make_dirs()
    existing = _count(NEG_DIR)
    print(f"\n{'='*60}")
    print("  NEGATIVE SAMPLE COLLECTION — Background noise")
    print(f"  Existing clips: {existing}")
    print(f"  Recording {n_clips} clips of 2s each")
    print("  Sources to include:")
    print("    - Silence / breathing")
    print("    - TV / YouTube playing")
    print("    - Fan or AC")
    print("    - Random conversation (talk about anything EXCEPT 'Hey Jarvis')")
    print("    - Keyboard typing")
    print("    - Music")
    print(f"{'='*60}\n")

    sources = [
        "Silence (just breathe normally)",
        "Say random words / sentences",
        "Hum or cough",
        "Play some music or TV in background",
        "Type on keyboard",
        "Random speech — 'What time is it', 'Okay Google', etc.",
        "Phone notifications / alerts",
    ]

    input("  Press ENTER when ready...")

    for i in range(n_clips):
        src = sources[i % len(sources)]
        print(f"\n  [{i+1}/{n_clips}] Source: {src}")
        print("  Recording...", end="", flush=True)

        audio = record_clip()
        path = _next_path(NEG_DIR, "negative")
        save_wav(path, audio)
        print(f" ✓ Saved")
        time.sleep(0.2)

    print(f"\n  Total negative clips: {_count(NEG_DIR)}")


def validate_dataset():
    """Print dataset statistics and quality report."""
    print(f"\n{'='*60}")
    print("  DATASET QUALITY REPORT")
    print(f"{'='*60}")

    for label, directory in [("POSITIVE", POS_DIR), ("NEGATIVE", NEG_DIR),
                               ("AUGMENTED", AUG_DIR), ("VALIDATION", VAL_DIR)]:
        clips = list(directory.glob("*.wav"))
        if not clips:
            print(f"\n  {label}: 0 clips")
            continue

        rms_vals = []
        durations = []
        for p in clips[:200]:  # sample first 200
            with wave.open(str(p), "rb") as wf:
                n_frames = wf.getnframes()
                rate     = wf.getframerate()
                raw      = wf.readframes(n_frames)
            audio    = np.frombuffer(raw, dtype=np.int16)
            rms_vals.append(rms(audio))
            durations.append(n_frames / rate)

        print(f"\n  {label} ({len(clips)} clips):")
        print(f"    RMS   mean={np.mean(rms_vals):.0f}  min={np.min(rms_vals):.0f}  max={np.max(rms_vals):.0f}")
        print(f"    Duration mean={np.mean(durations):.2f}s  min={np.min(durations):.2f}s")
        too_quiet = sum(1 for r in rms_vals if r < MIN_RMS)
        if too_quiet:
            print(f"    ⚠ {too_quiet} clips may be too quiet (RMS < {MIN_RMS})")

    pos = _count(POS_DIR)
    neg = _count(NEG_DIR)
    print(f"\n  RATIO — Positive:{pos} Negative:{neg}")
    if pos < 50:
        print("  ⚠ Need at least 50 positive clips for reasonable accuracy")
    if neg < pos * 2:
        print("  ⚠ Negative count should be at least 2× positive")
    if pos >= 200 and neg >= 400:
        print("  ✓ Dataset looks ready for training!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["positive", "negative", "validate"])
    parser.add_argument("--clips", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "positive":
        session_positive(args.clips)
    elif args.mode == "negative":
        session_negative(args.clips)
    elif args.mode == "validate":
        validate_dataset()

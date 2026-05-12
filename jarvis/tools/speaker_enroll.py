"""
tools/speaker_enroll.py — Owner Voice Enrollment Tool
======================================================
Records the owner's voice and creates a speaker profile.
After enrollment, only the owner's voice will activate JARVIS.

Usage:
    python tools/speaker_enroll.py enroll     # record 10 voice samples
    python tools/speaker_enroll.py test        # verify your voice matches
    python tools/speaker_enroll.py threshold   # tune similarity threshold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR    = Path(__file__).parent.parent
PROFILE_DIR = BASE_DIR / "models"
PROFILE_PATH = PROFILE_DIR / "owner_voice.npy"

SAMPLE_RATE  = 16_000
RECORD_SECS  = 3.0
N_ENROLL     = 10   # recordings during enrollment


def record_clip(duration: float = RECORD_SECS) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: pip install sounddevice")
        sys.exit(1)

    frames = []
    def cb(indata, f, t, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=1024, callback=cb):
        time.sleep(duration)

    return np.concatenate(frames).flatten()


def get_embedding(audio: np.ndarray, encoder) -> np.ndarray:
    """Convert int16 audio to Resemblyzer embedding."""
    f = audio.astype(np.float32) / 32768.0
    return encoder.embed_utterance(f)


def enroll():
    """Record N_ENROLL clips and save the averaged embedding as owner profile."""
    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        print("\n  ERROR: resemblyzer not installed.")
        print("  Run: pip install resemblyzer\n")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  OWNER VOICE ENROLLMENT")
    print("  Resemblyzer speaker verification")
    print(f"{'='*60}")
    print(f"\n  Recording {N_ENROLL} samples of {RECORD_SECS:.0f}s each.")
    print("  Say 'Hey JARVIS' or any natural speech after each prompt.")
    print("  Vary your tone slightly each time.\n")

    encoder = VoiceEncoder()
    input("  Press ENTER to begin...")

    embeddings = []
    for i in range(N_ENROLL):
        print(f"\n  [{i+1}/{N_ENROLL}] Speak now →", end="", flush=True)
        time.sleep(0.3)
        audio = record_clip()
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < 200:
            print(f" ✗ Too quiet (RMS={rms:.0f}) — try again")
            i -= 1
            continue
        emb = get_embedding(audio, encoder)
        embeddings.append(emb)
        print(f" ✓ OK (RMS={rms:.0f})")
        time.sleep(0.3)

    # Average all embeddings to create robust profile
    owner_embedding = np.mean(embeddings, axis=0)
    owner_embedding /= np.linalg.norm(owner_embedding)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(PROFILE_PATH), owner_embedding)

    print(f"\n  ✓ Owner profile saved: {PROFILE_PATH}")
    print("  Speaker verification is now ACTIVE.")
    print("  Only your voice will trigger JARVIS.\n")


def test_verification():
    """Verify your voice against the saved profile."""
    if not PROFILE_PATH.exists():
        print("\n  ERROR: No owner profile found. Run: python tools/speaker_enroll.py enroll\n")
        sys.exit(1)

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        print("  ERROR: pip install resemblyzer")
        sys.exit(1)

    encoder = VoiceEncoder()
    owner_emb = np.load(str(PROFILE_PATH))

    print(f"\n{'='*60}")
    print("  SPEAKER VERIFICATION TEST")
    print(f"{'='*60}")
    print("\n  Say something. Scores show how closely you match the enrolled voice.")
    print("  Press Ctrl+C to stop.\n")

    thresholds_tested = []

    try:
        while True:
            input("  Press ENTER to record →")
            audio = record_clip(duration=2.5)
            emb   = get_embedding(audio, encoder)
            sim   = float(np.dot(emb, owner_emb) /
                          (np.linalg.norm(emb) * np.linalg.norm(owner_emb) + 1e-8))
            thresholds_tested.append(sim)
            result = "✓ OWNER"  if sim >= 0.75 else "✗ REJECTED"
            bar    = "█" * int(sim * 40)
            print(f"  {result}  similarity={sim:.3f}  |{bar:<40}|")
    except KeyboardInterrupt:
        pass

    if thresholds_tested:
        print(f"\n  Your scores:  mean={np.mean(thresholds_tested):.3f}  "
              f"min={np.min(thresholds_tested):.3f}  max={np.max(thresholds_tested):.3f}")
        mean = np.mean(thresholds_tested)
        recommended = max(0.60, mean - 0.15)
        print(f"  Recommended threshold: {recommended:.2f}")
        print(f"  (Set SpeakerVerifier.THRESHOLD = {recommended:.2f} in engine.py)\n")


def tune_threshold():
    """Record multiple speakers to find the optimal threshold."""
    if not PROFILE_PATH.exists():
        print("  No owner profile. Run enroll first.")
        sys.exit(1)

    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        print("  pip install resemblyzer")
        sys.exit(1)

    encoder   = VoiceEncoder()
    owner_emb = np.load(str(PROFILE_PATH))

    print(f"\n{'='*60}")
    print("  THRESHOLD TUNING")
    print("  Test different speakers to find optimal threshold.")
    print("  'o' = you (owner)   'x' = someone else   'q' = quit")
    print(f"{'='*60}\n")

    owner_sims = []
    other_sims = []

    while True:
        who = input("  Who is speaking? [o=owner, x=other, q=quit] → ").strip().lower()
        if who == "q":
            break
        if who not in ("o", "x"):
            continue

        input("  Press ENTER to record →")
        audio = record_clip(duration=2.5)
        emb   = get_embedding(audio, encoder)
        sim   = float(np.dot(emb, owner_emb) /
                      (np.linalg.norm(emb) * np.linalg.norm(owner_emb) + 1e-8))

        if who == "o":
            owner_sims.append(sim)
            print(f"  OWNER  similarity={sim:.3f}")
        else:
            other_sims.append(sim)
            print(f"  OTHER  similarity={sim:.3f}")

    if owner_sims and other_sims:
        # Find threshold with max separation
        all_scores = [(s, 1) for s in owner_sims] + [(s, 0) for s in other_sims]
        best_t, best_acc = 0.7, 0
        for t in np.arange(0.5, 0.95, 0.01):
            correct = sum(
                1 for s, label in all_scores
                if (s >= t) == (label == 1)
            )
            acc = correct / len(all_scores)
            if acc > best_acc:
                best_acc, best_t = acc, t

        print(f"\n  Owner  mean={np.mean(owner_sims):.3f}  min={np.min(owner_sims):.3f}")
        print(f"  Other  mean={np.mean(other_sims):.3f}  max={np.max(other_sims):.3f}")
        print(f"\n  ✓ Optimal threshold: {best_t:.2f}  (accuracy={best_acc:.1%})")
        print(f"  Set SpeakerVerifier.THRESHOLD = {best_t:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["enroll", "test", "threshold"])
    args = parser.parse_args()

    if args.mode == "enroll":
        enroll()
    elif args.mode == "test":
        test_verification()
    elif args.mode == "threshold":
        tune_threshold()

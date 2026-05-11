"""
wake/enroll.py — Speaker Enrollment CLI
=========================================
Run this once to enroll your voice so JARVIS only responds to you.

Usage:
    python -m wake.enroll
    python -m wake.enroll --from-files path/to/*.wav
    python -m wake.enroll --test          # verify against stored embedding
    python -m wake.enroll --threshold 0.75
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

from .speaker import SpeakerVerifier
from .config import SPEAKER_SIMILARITY_MIN


def enroll(n_samples: int, seconds_each: float):
    sv = SpeakerVerifier()
    if not sv.available:
        print("\n  ✗ Resemblyzer not installed.")
        print("  Run: pip install resemblyzer\n")
        sys.exit(1)
    ok = sv.enroll_from_mic(n_samples=n_samples, seconds_each=seconds_each)
    if ok:
        print("  ✓ Speaker enrollment complete!")
        print("  Update config.py: SPEAKER_VERIFY_ENABLED = True\n")
    else:
        print("  ✗ Enrollment failed.\n")


def enroll_from_files(paths: list):
    sv = SpeakerVerifier()
    ok = sv.enroll_from_files(paths)
    if ok:
        print("  ✓ Enrolled from files!")
    else:
        print("  ✗ Failed to enroll from files.")


def test_verification(threshold: float):
    sv = SpeakerVerifier()
    if not sv.enrolled:
        print("  ✗ No enrolled voice. Run enrollment first.\n")
        return

    try:
        import sounddevice as sd
    except ImportError:
        print("  Install: pip install sounddevice")
        return

    print(f"\n  Testing speaker verification (threshold={threshold})")
    print("  Speak for 3 seconds when prompted...\n")

    from .config import SAMPLE_RATE, SPEAKER_ENROLL_SECONDS
    input("  Press Enter and speak...")
    recording = sd.rec(
        int(SAMPLE_RATE * 3.0),
        samplerate=SAMPLE_RATE, channels=1, dtype="int16"
    )
    sd.wait()

    import numpy as np
    audio_np = recording.flatten()
    sv.set_threshold(threshold)
    accepted, score = sv.verify(audio_np)

    print(f"\n  Similarity score: {score:.3f}")
    print(f"  Threshold:        {threshold:.3f}")
    print(f"  Result:           {'✓ ACCEPTED' if accepted else '✗ REJECTED'}\n")


def main():
    parser = argparse.ArgumentParser(description="JARVIS speaker enrollment")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of voice samples to record")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Duration of each sample")
    parser.add_argument("--from-files", nargs="+", metavar="WAV",
                        help="Enroll from existing WAV files instead of mic")
    parser.add_argument("--test", action="store_true",
                        help="Test verification against stored embedding")
    parser.add_argument("--threshold", type=float, default=SPEAKER_SIMILARITY_MIN,
                        help="Similarity threshold for acceptance")
    args = parser.parse_args()

    if args.test:
        test_verification(args.threshold)
    elif args.from_files:
        enroll_from_files([Path(p) for p in args.from_files])
    else:
        enroll(n_samples=args.n_samples, seconds_each=args.seconds)


if __name__ == "__main__":
    main()

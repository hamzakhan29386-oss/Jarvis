"""
tools/false_positive_test.py — False Positive Benchmark
=========================================================
Runs the detector against background audio files and reports
how many times it would have falsely triggered.

Use this BEFORE going to production to validate your threshold.

Usage:
    python tools/false_positive_test.py --audio dataset/negative --duration 60
    python tools/false_positive_test.py --threshold 0.5 --window 3
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


def load_wav_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        rate = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    if rate != 16_000:
        n_out = int(len(audio) * 16_000 / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)), audio
        ).astype(np.int16)
    return audio


def run_fp_test(args):
    from wake.engine import WakeEngine, EngineConfig

    config = EngineConfig(
        threshold      = args.threshold,
        rolling_window = args.window,
        cooldown_secs  = 2.5,
        model_path     = args.model or None,
        debug_scores   = False,
    )

    engine = WakeEngine(config=config)
    if not engine.ready:
        print("\n  ERROR: Model failed to load.\n")
        sys.exit(1)

    detector = engine._detector

    # Load negative clips
    neg_dir = BASE_DIR / args.audio
    clips   = list(neg_dir.glob("*.wav"))

    if not clips:
        print(f"\n  No WAV files in {neg_dir}\n")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FALSE POSITIVE BENCHMARK")
    print(f"  Threshold: {args.threshold}  Window: {args.window}")
    print(f"  Clips: {len(clips)} from {neg_dir}")
    print(f"{'='*60}\n")

    CHUNK  = 1280
    total_chunks     = 0
    total_triggers   = 0
    total_secs       = 0.0
    last_trigger_chunk = -1000
    cooldown_chunks    = int(2.5 * 16_000 / CHUNK)

    for i, clip_path in enumerate(clips[:args.max_clips]):
        try:
            audio = load_wav_16k_mono(clip_path)
        except Exception as e:
            continue

        clip_secs = len(audio) / 16_000
        total_secs += clip_secs
        clip_triggers = 0

        for start in range(0, len(audio) - CHUNK, CHUNK):
            chunk = audio[start:start + CHUNK]
            if len(chunk) < CHUNK:
                chunk = np.pad(chunk, (0, CHUNK - len(chunk)))

            score = detector.predict(chunk)
            total_chunks += 1

            if score >= args.threshold:
                if total_chunks - last_trigger_chunk > cooldown_chunks:
                    clip_triggers   += 1
                    total_triggers  += 1
                    last_trigger_chunk = total_chunks

        status = "⚠ TRIGGERED" if clip_triggers > 0 else "✓ clean"
        if (i + 1) % 10 == 0 or clip_triggers > 0:
            print(f"  [{i+1:3d}] {clip_path.name:<40} {status} ({clip_triggers})")

    total_mins = total_secs / 60
    fpr_per_min = total_triggers / max(total_mins, 0.01)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Audio processed:    {total_secs:.0f}s  ({total_mins:.1f} min)")
    print(f"  Total chunks:       {total_chunks}")
    print(f"  False triggers:     {total_triggers}")
    print(f"  False positive rate: {fpr_per_min:.2f} / minute")

    if fpr_per_min < 0.1:
        print(f"\n  ✓ EXCELLENT — production ready (< 0.1/min)")
    elif fpr_per_min < 0.5:
        print(f"\n  ✓ GOOD — acceptable for most use cases (< 0.5/min)")
    elif fpr_per_min < 2.0:
        print(f"\n  ⚠ MODERATE — consider raising threshold or increasing rolling window")
    else:
        print(f"\n  ✗ HIGH — too many false positives, raise threshold or retrain")
        print(f"     Try: --threshold {min(0.95, args.threshold + 0.1):.2f}")

    print(f"\n  Recommendation: threshold={args.threshold}  window={args.window}")
    if fpr_per_min > 0.5:
        new_t = min(0.95, args.threshold + 0.05)
        new_w = min(5, args.window + 1)
        print(f"  Try:            threshold={new_t:.2f}  window={new_w}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",     default="dataset/negative")
    parser.add_argument("--model",     default=None)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--window",    type=int,   default=3)
    parser.add_argument("--max-clips", dest="max_clips", type=int, default=200)
    args = parser.parse_args()
    run_fp_test(args)

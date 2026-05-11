"""
tools/threshold_tuner.py — Live Wake Word Confidence Visualizer
================================================================
Runs the wake engine in debug mode so you can watch confidence
scores in real-time and find your optimal threshold.

Shows:
  - Live scrolling confidence bar
  - Detection events
  - Audio RMS level
  - Suggested threshold based on your patterns

Usage:
    python tools/threshold_tuner.py
    python tools/threshold_tuner.py --model models/hey_jarvis_custom.onnx
    python tools/threshold_tuner.py --threshold 0.4 --window 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from wake.engine import WakeEngine, EngineConfig, DetectionEvent


HISTORY_LEN = 60   # seconds of history to display


def run_tuner(args):
    os.makedirs(BASE_DIR / "logs", exist_ok=True)

    # ── Engine setup ─────────────────────────────────────────────────────
    config = EngineConfig(
        threshold      = args.threshold,
        rolling_window = args.window,
        cooldown_secs  = 1.5,
        model_path     = args.model,
        debug_scores   = False,   # we handle display ourselves
    )

    engine = WakeEngine(config=config)

    if not engine.ready:
        print("\n  ERROR: Model failed to load.")
        print("  Check: pip install openwakeword")
        print("  Or specify: --model path/to/model.onnx\n")
        sys.exit(1)

    # ── State ─────────────────────────────────────────────────────────────
    score_history   = deque(maxlen=HISTORY_LEN * 10)   # ~100ms chunks
    detections      = []
    peaks_above_bg  = []
    last_score      = 0.0
    lock            = threading.Lock()

    def on_score(score: float):
        nonlocal last_score
        with lock:
            last_score = score
            score_history.append((time.time(), score))

    def on_wake(event: DetectionEvent):
        with lock:
            detections.append(event)
            print(f"\n\n  ★ WAKE DETECTED  confidence={event.confidence:.3f}  "
                  f"rms={event.audio_rms:.0f}  latency={event.latency_ms:.1f}ms\n")

    engine.on_score = on_score
    engine.on_wake  = on_wake
    engine.start()

    # ── Display loop ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  JARVIS Wake Word — Live Confidence Monitor")
    print(f"  Model:     {engine._detector._model_key}")
    print(f"  Threshold: {config.threshold:.2f}  |  Window: {config.rolling_window}")
    print(f"  Press Ctrl+C to stop and see recommendations")
    print(f"{'='*64}")
    print(f"\n  Say 'Hey JARVIS' several times, then stay quiet for noise floor.\n")
    print(f"  {'Score':<8} {'Bar':<42} {'Alert'}")
    print(f"  {'-'*7}  {'-'*41}  {'-'*10}")

    all_scores      = []
    wake_scores     = []
    background_scores = []

    try:
        t_phase_start = time.time()
        phase = "wake"  # start by asking user to say wake word

        while True:
            time.sleep(0.10)
            with lock:
                score = last_score

            all_scores.append(score)
            bar_len = int(score * 42)
            bar     = "█" * bar_len + "░" * (42 - bar_len)

            # Color-code: above threshold = red, near = yellow, low = green
            if score >= config.threshold:
                label = "◄ DETECTED"
                wake_scores.append(score)
            elif score >= config.threshold * 0.7:
                label = "▲ near"
            else:
                label = ""
                background_scores.append(score)

            print(f"\r  {score:.4f}  |{bar}| {label:<12}", end="", flush=True)

    except KeyboardInterrupt:
        engine.stop()
        print("\n")

    # ── Post-session analysis ─────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  SESSION ANALYSIS")
    print(f"{'='*64}")
    print(f"  Total frames:      {len(all_scores)}")
    print(f"  Detections:        {len(detections)}")

    if all_scores:
        print(f"  Score range:       {min(all_scores):.4f} – {max(all_scores):.4f}")
        print(f"  Score mean:        {sum(all_scores)/len(all_scores):.4f}")

    if wake_scores:
        mean_wake = sum(wake_scores) / len(wake_scores)
        print(f"\n  Wake word scores:  {min(wake_scores):.3f} – {max(wake_scores):.3f}  "
              f"(mean={mean_wake:.3f})")

    if background_scores:
        mean_bg = sum(background_scores) / len(background_scores)
        print(f"  Background scores: {min(background_scores):.3f} – {max(background_scores):.3f}  "
              f"(mean={mean_bg:.3f})")

    # Recommend threshold
    if wake_scores and background_scores:
        mean_wake = sum(wake_scores) / len(wake_scores)
        max_bg    = max(background_scores)
        # Set threshold at 60% of the gap between bg ceiling and wake floor
        wake_floor  = min(wake_scores)
        gap         = wake_floor - max_bg
        recommended = max_bg + gap * 0.6

        print(f"\n  ✓ Recommended threshold: {recommended:.2f}")
        print(f"  (Currently: {config.threshold:.2f})")
        print(f"\n  Update EngineConfig(threshold={recommended:.2f}) in your server code.")
    elif not wake_scores:
        print(f"\n  ⚠ No wake words detected above threshold {config.threshold:.2f}")
        print(f"  Try lowering: --threshold {config.threshold * 0.7:.2f}")
    elif not background_scores:
        print(f"\n  ⚠ No quiet periods recorded — can't estimate noise floor")

    print(f"\n  False activation rate: {len(detections) / max(1, len(all_scores)/10):.3f}/min")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live wake word threshold tuner")
    parser.add_argument("--model",     default=None,  help="ONNX model path (default: built-in)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Detection threshold")
    parser.add_argument("--window",    type=int,   default=1,    help="Rolling window size")
    parser.add_argument("--device",    type=int,   default=None, help="Microphone device index")
    args = parser.parse_args()

    run_tuner(args)

"""
tools/mic_diagnostics.py — Microphone Health Checker
=====================================================
Validates your audio setup before running the wake engine.
Catches the most common failure modes:
  - Wrong sample rate
  - Stereo vs mono mismatch
  - Silent microphone
  - Too-quiet input
  - Incorrect dtype

Usage:
    python tools/mic_diagnostics.py
    python tools/mic_diagnostics.py --device 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


def run_diagnostics(device_index: int = None):
    print(f"\n{'='*60}")
    print("  JARVIS Microphone Diagnostics")
    print(f"{'='*60}\n")

    # ── 1. Check sounddevice ─────────────────────────────────────────────
    print("  [1/6] Checking sounddevice...")
    try:
        import sounddevice as sd
        print(f"        ✓ sounddevice {sd.__version__}")
    except ImportError:
        print("        ✗ Not installed: pip install sounddevice")
        sys.exit(1)

    # ── 2. List devices ──────────────────────────────────────────────────
    print("\n  [2/6] Available audio devices:")
    try:
        import numpy as np
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                marker = " ◄ DEFAULT" if i == sd.default.device[0] else ""
                marker += " ◄ SELECTED" if i == device_index else ""
                print(f"        [{i:2d}] {dev['name'][:40]:<40} "
                      f"ch={dev['max_input_channels']}  "
                      f"rate={int(dev['default_samplerate'])}Hz{marker}")
    except Exception as e:
        print(f"        Could not list devices: {e}")

    # ── 3. Test device open ──────────────────────────────────────────────
    TARGET_RATE = 16_000
    print(f"\n  [3/6] Opening microphone at {TARGET_RATE}Hz (mono, int16)...")
    try:
        frames = []
        def cb(indata, f, t, status):
            if status:
                print(f"        ⚠ Status: {status}")
            frames.append(indata.copy())

        with sd.InputStream(
            device=device_index,
            samplerate=TARGET_RATE,
            channels=1,
            dtype="int16",
            blocksize=1280,
            callback=cb,
        ):
            time.sleep(1.0)

        import numpy as np
        audio = np.concatenate(frames).flatten()
        print(f"        ✓ Stream opened. Captured {len(audio)} samples.")
    except Exception as e:
        print(f"        ✗ FAILED: {e}")
        print(f"\n        If this is a sample-rate error, your device may not support {TARGET_RATE}Hz.")
        print(f"        Check supported rates and use src_rate parameter in engine.start().")
        sys.exit(1)

    # ── 4. Audio format check ────────────────────────────────────────────
    print("\n  [4/6] Validating audio format...")
    print(f"        dtype:    {audio.dtype}  (expected int16) {'✓' if audio.dtype == np.int16 else '✗'}")
    print(f"        shape:    {audio.shape}  (expected 1D) {'✓' if audio.ndim == 1 else '✗'}")
    print(f"        samples:  {len(audio)} (~{len(audio)/TARGET_RATE*1000:.0f}ms)")
    print(f"        rate:     {TARGET_RATE}Hz ✓")

    # ── 5. Signal level check ────────────────────────────────────────────
    print("\n  [5/6] Signal level check (stay quiet)...")
    rms_quiet = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    peak = int(np.max(np.abs(audio)))

    print(f"        RMS:  {rms_quiet:.1f}  (silence should be < 400)")
    print(f"        Peak: {peak}  (max 32767)")

    if rms_quiet < 50:
        print("        ✓ Very quiet — microphone is silent (good noise floor)")
    elif rms_quiet < 400:
        print("        ✓ Acceptable noise floor")
    elif rms_quiet < 1500:
        print("        ⚠ Moderately noisy — fan/AC? Wake word may have false positives")
    else:
        print("        ✗ Very noisy — check microphone placement or use noise cancellation")

    # Now record with speech
    print("\n  [6/6] Speech level test — say 'Hey JARVIS' now →", end="", flush=True)
    frames2 = []
    def cb2(indata, f, t, status):
        frames2.append(indata.copy())

    with sd.InputStream(device=device_index, samplerate=TARGET_RATE,
                        channels=1, dtype="int16", blocksize=1280, callback=cb2):
        time.sleep(2.5)

    audio2 = np.concatenate(frames2).flatten()
    rms_speech = float(np.sqrt(np.mean(audio2.astype(np.float32) ** 2)))
    snr_estimate = rms_speech / (rms_quiet + 1e-6)

    print(f"\n        Speech RMS:  {rms_speech:.1f}")
    print(f"        SNR estimate: {20 * np.log10(snr_estimate):.1f} dB")

    if rms_speech < 500:
        print("        ✗ Speech too quiet — move closer to mic or increase input gain")
    elif rms_speech < 2000:
        print("        ⚠ Speech a bit quiet — wake detection may be unreliable from distance")
    else:
        print("        ✓ Good speech level")

    if snr_estimate > 5:
        print("        ✓ Good SNR — wake word should be detectable")
    else:
        print("        ✗ Low SNR — noise suppression strongly recommended")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")

    issues = []
    if rms_quiet > 1500:
        issues.append("High background noise — use noise suppression")
    if rms_speech < 500:
        issues.append("Speech level too low — increase mic gain")
    if snr_estimate < 3:
        issues.append("Poor SNR — microphone placement needs improvement")

    if not issues:
        print("  ✓ All checks passed! Your microphone setup looks good.\n")
        print("  Recommended engine settings:")
        print(f"    rms_gate  = {max(50, int(rms_quiet * 2))}")
        print(f"    threshold = 0.35 (tune with tools/threshold_tuner.py)")
    else:
        print("  Issues found:")
        for i in issues:
            print(f"    ✗ {i}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None, help="Microphone device index")
    args = parser.parse_args()
    run_diagnostics(args.device)

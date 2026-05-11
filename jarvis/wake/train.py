"""
wake/train.py — Custom Wake Word Training Pipeline
====================================================
Complete pipeline to create a personalized "Hey JARVIS" model:

    Step 1: dataset/     ← collect or download audio clips
    Step 2: augment      ← pitch shift, speed, noise, room simulation
    Step 3: train        ← openwakeword custom model training
    Step 4: evaluate     ← measure false positives / true positives
    Step 5: deploy       ← update config.py with new model path

Run:
    python -m wake.train collect    → guided mic recording session
    python -m wake.train augment    → augment collected clips
    python -m wake.train evaluate   → score existing model on your dataset
    python -m wake.train all        → full pipeline

Requirements:
    pip install openwakeword sounddevice soundfile audiomentations
"""

import argparse
import logging
import os
import sys
import time
import threading
import wave
from pathlib import Path

import numpy as np

from .config import (
    SAMPLE_RATE, WAKE_THRESHOLD, WAKE_MODELS,
    DATASET_DIR, MODELS_DIR, CHUNK_SAMPLES,
)

log = logging.getLogger("jarvis.wake.train")

# ── Dataset structure ─────────────────────────────────────────────────────────
# dataset/
#   positive/    ← "Hey JARVIS" recordings
#   negative/    ← background noise, random speech, music, etc.
#   augmented/
#     positive/  ← pitch-shifted, speed-varied, noisy versions
#     negative/

POSITIVE_DIR  = DATASET_DIR / "positive"
NEGATIVE_DIR  = DATASET_DIR / "negative"
AUG_POS_DIR   = DATASET_DIR / "augmented" / "positive"
AUG_NEG_DIR   = DATASET_DIR / "augmented" / "negative"

TARGET_POS    = 300   # target number of positive samples (augmented)
TARGET_NEG    = 1000  # target number of negative samples
MIN_RAW_POS   = 20    # minimum raw recordings before augmenting


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_positive_samples(n: int = 50, seconds: float = 2.0):
    """
    Guided mic recording: captures N "Hey JARVIS" clips.
    Gives audio feedback between each take.

    Args:
        n: Number of recordings to collect.
        seconds: Length of each recording.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit("Install: pip install sounddevice soundfile")

    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(POSITIVE_DIR.glob("*.wav")))

    print("\n" + "═" * 60)
    print("  POSITIVE SAMPLE COLLECTION — Say 'Hey JARVIS'")
    print("  Vary your tone, speed, distance, and room each time.")
    print(f"  Target: {n} clips   Already have: {existing}")
    print("═" * 60 + "\n")

    prompts = [
        "Normal tone", "Slightly louder", "Slightly quieter",
        "Faster", "Slower", "From 2 meters away", "In a whisper",
        "After a pause", "Different room accent", "Emphasise JARVIS",
    ]

    for i in range(n):
        tip = prompts[i % len(prompts)]
        input(f"  [{i+1:03d}/{n}] Tip: {tip} — press Enter then speak...")

        # Brief 300ms tone so you know when recording starts
        _play_tone(880, 0.15)

        recording = sd.rec(
            int(SAMPLE_RATE * seconds),
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        sd.wait()

        path = POSITIVE_DIR / f"pos_{existing + i + 1:04d}.wav"
        sf.write(str(path), recording, SAMPLE_RATE, subtype="PCM_16")
        print(f"    ✓ Saved → {path.name}")

    print(f"\n  Done! {n} clips saved to {POSITIVE_DIR}\n")


def collect_negative_samples(n: int = 50, seconds: float = 3.0):
    """
    Guided mic recording of background noise for negative examples.
    Run in various environments: quiet room, TV on, music, typing, etc.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        sys.exit("Install: pip install sounddevice soundfile")

    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(NEGATIVE_DIR.glob("*.wav")))

    print("\n" + "═" * 60)
    print("  NEGATIVE SAMPLE COLLECTION")
    print("  Record ambient sound, speech, TV, typing — NOT 'Hey JARVIS'")
    print("═" * 60 + "\n")

    neg_prompts = [
        "Silence — just ambient room noise",
        "Keyboard typing",
        "TV or radio in background",
        "Normal conversation (say anything except the wake word)",
        "Music playing",
        "Fan or HVAC noise",
        "Similar phrases: 'Hey Google', 'Hey Siri', 'OK Google'",
    ]

    for i in range(n):
        prompt = neg_prompts[i % len(neg_prompts)]
        input(f"  [{i+1:03d}/{n}] Scene: {prompt} — press Enter when ready...")

        _play_tone(440, 0.1)
        recording = sd.rec(
            int(SAMPLE_RATE * seconds),
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        sd.wait()

        path = NEGATIVE_DIR / f"neg_{existing + i + 1:04d}.wav"
        sf.write(str(path), recording, SAMPLE_RATE, subtype="PCM_16")
        print(f"    ✓ Saved → {path.name}")

    print(f"\n  Done! {n} clips saved to {NEGATIVE_DIR}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment_dataset(
    target_positive: int = TARGET_POS,
    target_negative: int = TARGET_NEG,
):
    """
    Augment raw clips to hit the target count using audiomentations.
    Transformations applied:
        - Pitch shift (±3 semitones)
        - Time stretch (0.85x – 1.15x)
        - Gaussian noise injection
        - Room impulse response (reverb)
        - Volume gain adjustment
    """
    try:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, RoomSimulator
        import soundfile as sf
    except ImportError:
        print("  Install: pip install audiomentations soundfile")
        print("  Skipping augmentation — raw clips only.")
        return

    AUG_POS_DIR.mkdir(parents=True, exist_ok=True)
    AUG_NEG_DIR.mkdir(parents=True, exist_ok=True)

    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.7),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5),
    ])

    def _augment_folder(src: Path, dst: Path, target: int, label: str):
        sources = sorted(src.glob("*.wav"))
        if not sources:
            print(f"  ✗ No WAV files in {src}")
            return

        # First copy originals
        count = 0
        for wav in sources:
            audio, sr = sf.read(str(wav), dtype="float32")
            out_path = dst / f"{label}_{count:04d}_orig.wav"
            sf.write(str(out_path), audio, sr, subtype="PCM_16")
            count += 1

        # Then augment until target reached
        while count < target:
            src_wav = sources[count % len(sources)]
            audio, sr = sf.read(str(src_wav), dtype="float32")
            augmented = augment(samples=audio, sample_rate=sr)
            out_path = dst / f"{label}_{count:04d}_aug.wav"
            sf.write(str(out_path), augmented, sr, subtype="PCM_16")
            count += 1

        print(f"  ✓ {label}: {count} clips in {dst}")

    raw_pos = len(list(POSITIVE_DIR.glob("*.wav")))
    raw_neg = len(list(NEGATIVE_DIR.glob("*.wav")))

    if raw_pos < MIN_RAW_POS:
        print(f"\n  ⚠ Only {raw_pos} positive clips (minimum {MIN_RAW_POS}).")
        print("  Run: python -m wake.train collect --positive\n")
        return

    print(f"\n  Augmenting {raw_pos} positive → {target_positive} total...")
    _augment_folder(POSITIVE_DIR, AUG_POS_DIR, target_positive, "pos")

    print(f"  Augmenting {raw_neg} negative → {target_negative} total...")
    _augment_folder(NEGATIVE_DIR, AUG_NEG_DIR, target_negative, "neg")

    print("\n  Augmentation complete!\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model_name: str = "hey_jarvis_custom", epochs: int = 150):
    """
    Train a custom openwakeword model on the collected dataset.

    Uses augmented clips if available, falls back to raw clips.
    Saves model as <model_name>.onnx in the models/ directory.

    Args:
        model_name: Name for the output model file (without .onnx).
        epochs: Training epochs (more = better, slower).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Choose best available clips
    pos_dir = AUG_POS_DIR if any(AUG_POS_DIR.glob("*.wav")) else POSITIVE_DIR
    neg_dir = AUG_NEG_DIR if any(AUG_NEG_DIR.glob("*.wav")) else NEGATIVE_DIR

    n_pos = len(list(pos_dir.glob("*.wav")))
    n_neg = len(list(neg_dir.glob("*.wav")))

    print(f"\n  Training '{model_name}'")
    print(f"  Positive: {n_pos} clips from {pos_dir}")
    print(f"  Negative: {n_neg} clips from {neg_dir}")
    print(f"  Epochs:   {epochs}")
    print(f"  Output:   {MODELS_DIR / (model_name + '.onnx')}\n")

    if n_pos < 20:
        print(f"  ✗ Need at least 20 positive clips (have {n_pos}). Run collect first.")
        return False

    try:
        from openwakeword.custom import train_custom_model
        train_custom_model(
            positive_reference_clips=[str(pos_dir)],
            negative_reference_clips=[str(neg_dir)],
            output_dir=str(MODELS_DIR),
            model_name=model_name,
            epochs=epochs,
            target_false_positive_rate=0.005,
        )
        print(f"\n  ✓ Model saved: {MODELS_DIR / (model_name + '.onnx')}")
        print(f"\n  Update config.py:")
        print(f'    WAKE_MODELS = ["{MODELS_DIR / (model_name + ".onnx")}"]')
        return True
    except AttributeError:
        _try_notebook_training(model_name)
        return False
    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        return False


def _try_notebook_training(model_name: str):
    print("\n  openwakeword's train_custom_model is not available in this version.")
    print("  Use the official training notebook instead:\n")
    print("  https://github.com/dscripka/openWakeWord/blob/main/notebooks/automated_model_training.ipynb\n")
    print(f"  Point it at:")
    print(f"    Positive: {AUG_POS_DIR}")
    print(f"    Negative: {AUG_NEG_DIR}")
    print(f"    Output:   {MODELS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_path: str = None, threshold: float = WAKE_THRESHOLD):
    """
    Evaluate the model on the local dataset.
    Reports: True Positive Rate, False Positive Rate, recommended threshold.

    Args:
        model_path: Path to .onnx model. Defaults to the first WAKE_MODELS entry.
        threshold: Detection threshold for evaluation.
    """
    try:
        from openwakeword.model import Model
        import soundfile as sf
    except ImportError:
        print("  Install: pip install openwakeword soundfile")
        return

    model_to_test = model_path or WAKE_MODELS[0]
    try:
        oww = Model(wakeword_models=[str(model_to_test)], inference_framework="onnx")
    except Exception as e:
        print(f"  ✗ Could not load model {model_to_test}: {e}")
        return

    def _score_folder(folder: Path, expected_positive: bool) -> dict:
        tp = fp = tn = fn = 0
        scores = []
        for wav in sorted(folder.glob("*.wav")):
            try:
                audio, sr = sf.read(str(wav), dtype="float32")
                if sr != SAMPLE_RATE:
                    print(f"  Skip (wrong SR): {wav.name}")
                    continue
                audio_i16 = (audio * 32767).astype(np.int16)
                # Feed in CHUNK_SAMPLES chunks
                peak = 0.0
                for offset in range(0, len(audio_i16) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                    chunk = audio_i16[offset:offset + CHUNK_SAMPLES]
                    preds = oww.predict(chunk)
                    s = max(preds.values(), default=0.0)
                    peak = max(peak, float(s))
                scores.append(peak)
                detected = peak >= threshold
                if expected_positive:
                    if detected: tp += 1
                    else: fn += 1
                else:
                    if detected: fp += 1
                    else: tn += 1
            except Exception as e:
                log.debug(f"[Eval] {wav.name}: {e}")
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "scores": scores}

    print(f"\n  Evaluating: {model_to_test} @ threshold={threshold}\n")

    pos_dir = AUG_POS_DIR if any(AUG_POS_DIR.glob("*.wav")) else POSITIVE_DIR
    neg_dir = AUG_NEG_DIR if any(AUG_NEG_DIR.glob("*.wav")) else NEGATIVE_DIR

    print(f"  Positive clips: {pos_dir} ({len(list(pos_dir.glob('*.wav')))} files)")
    r_pos = _score_folder(pos_dir, expected_positive=True)

    print(f"  Negative clips: {neg_dir} ({len(list(neg_dir.glob('*.wav')))} files)")
    r_neg = _score_folder(neg_dir, expected_positive=False)

    tpr = r_pos["tp"] / max(r_pos["tp"] + r_pos["fn"], 1)
    fpr = r_neg["fp"] / max(r_neg["fp"] + r_neg["tn"], 1)

    pos_scores = r_pos["scores"]
    neg_scores = r_neg["scores"]

    print("\n  ─── Results ─────────────────────────────────────────")
    print(f"  True  Positive Rate (TPR):  {tpr:.1%}  ({r_pos['tp']}/{r_pos['tp']+r_pos['fn']})")
    print(f"  False Positive Rate (FPR):  {fpr:.1%}  ({r_neg['fp']}/{r_neg['fp']+r_neg['tn']})")

    if pos_scores:
        print(f"\n  Positive score stats:")
        print(f"    mean={np.mean(pos_scores):.3f}  min={np.min(pos_scores):.3f}  max={np.max(pos_scores):.3f}")
    if neg_scores:
        print(f"  Negative score stats:")
        print(f"    mean={np.mean(neg_scores):.3f}  min={np.min(neg_scores):.3f}  max={np.max(neg_scores):.3f}")

    # Threshold recommendation: midpoint between mean negative and mean positive
    if pos_scores and neg_scores:
        recommended = (np.mean(pos_scores) + np.mean(neg_scores)) / 2
        print(f"\n  Recommended threshold ≈ {recommended:.2f}")
        print(f"  Current threshold      = {threshold:.2f}")
        if tpr < 0.90:
            print("  ⚠ TPR below 90% — consider lowering threshold or collecting more data")
        if fpr > 0.05:
            print("  ⚠ FPR above 5% — consider raising threshold or adding more negative samples")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _play_tone(freq: float, duration: float):
    """Play a brief audio cue using sounddevice."""
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 0.3 * 32767).astype(np.int16)
        sd.play(wave, SAMPLE_RATE)
        sd.wait()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="JARVIS wake word training tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect        Guided mic recording session
  augment        Augment collected clips
  train          Train custom ONNX model
  evaluate       Evaluate model on your dataset
  all            Full pipeline: collect → augment → train → evaluate
        """,
    )
    parser.add_argument("command", choices=["collect", "augment", "train", "evaluate", "all"])
    parser.add_argument("--model-name", default="hey_jarvis_custom")
    parser.add_argument("--positive", action="store_true", help="collect only positive samples")
    parser.add_argument("--negative", action="store_true", help="collect only negative samples")
    parser.add_argument("--n", type=int, default=50, help="number of clips to record")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=WAKE_THRESHOLD)
    args = parser.parse_args()

    if args.command in ("collect", "all"):
        if not args.negative:
            collect_positive_samples(n=args.n)
        if not args.positive:
            collect_negative_samples(n=args.n)

    if args.command in ("augment", "all"):
        augment_dataset()

    if args.command in ("train", "all"):
        train_model(model_name=args.model_name, epochs=args.epochs)

    if args.command in ("evaluate", "all"):
        evaluate_model(threshold=args.threshold)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    main()

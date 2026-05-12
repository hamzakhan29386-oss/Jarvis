"""
wake/train.py — Production Wake Word Training Pipeline
========================================================
Trains a custom "Hey JARVIS" wake word model using OpenWakeWord.

PATH RESOLUTION (fixed):
    This script resolves all dataset paths relative to its OWN location
    (the wake/ directory), not relative to cwd. So it works correctly
    whether you run it as:
        python -m wake.train          (from jarvis/)
        python train.py               (from jarvis/wake/)
        python wake/train.py          (from jarvis/)

    Dataset must be at:   <this file's directory>/dataset/positive/
    i.e.  jarvis/wake/dataset/positive/*.wav

Usage:
    cd C:\\Users\\hamza\\Desktop\\Jarvis-2\\jarvis
    python -m wake.train --epochs 150

    Or with explicit paths:
    python -m wake.train --positive C:\\path\\to\\positive --negative C:\\path\\to\\negative
"""

import argparse
import logging
import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="  [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("jarvis.wake.train")

# ── THIS FILE'S DIRECTORY (always jarvis/wake/) ───────────────────────────────
WAKE_DIR = Path(__file__).resolve().parent          # jarvis/wake/
DATASET_DIR = WAKE_DIR / "dataset"                  # jarvis/wake/dataset/
MODELS_DIR  = WAKE_DIR / "models"                   # jarvis/wake/models/

# ── Default dataset layout (relative to WAKE_DIR, not cwd) ───────────────────
DEFAULT_POSITIVE    = [DATASET_DIR / "positive", DATASET_DIR / "augmented"]
DEFAULT_NEGATIVE    = [DATASET_DIR / "negative"]
DEFAULT_VALIDATION  = DATASET_DIR / "validation"
DEFAULT_OUTPUT      = MODELS_DIR / "hey_jarvis_custom.onnx"

# Audio constants
REQUIRED_SR     = 16000
REQUIRED_BITS   = 16
REQUIRED_CH     = 1
CHUNK_SAMPLES   = 1280          # 80 ms @ 16 kHz — required by OpenWakeWord
MIN_CLIP_SECS   = 0.5
MAX_CLIP_SECS   = 4.0


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 62)
    print("  JARVIS Wake Word Training Pipeline")
    print("=" * 62)
    print(f"  Script location : {WAKE_DIR}")
    print(f"  Working dir     : {Path.cwd()}")
    print(f"  Dataset root    : {DATASET_DIR}")
    print(f"  Models output   : {MODELS_DIR}")
    print("=" * 62 + "\n")


def diagnose_paths(positive_dirs: list[Path], negative_dirs: list[Path]):
    """Print a clear picture of what exists and what's missing."""
    print("  [DIAGNOSIS] Scanning dataset directories...\n")

    for role, dirs in [("POSITIVE", positive_dirs), ("NEGATIVE", negative_dirs)]:
        for d in dirs:
            exists = d.exists()
            if exists:
                wavs = list(d.glob("*.wav")) + list(d.glob("**/*.wav"))
                print(f"  {'✔' if wavs else '⚠'} {role} {d}")
                print(f"      → {len(wavs)} WAV files found")
                if wavs:
                    # Sample format of first file
                    _check_wav_format(wavs[0], verbose=True)
            else:
                print(f"  ✖ {role} {d}")
                print(f"      → DIRECTORY DOES NOT EXIST")
    print()


def _check_wav_format(path: Path, verbose: bool = False) -> dict:
    """Validate WAV file format. Returns info dict."""
    info = {"ok": False, "path": str(path)}
    try:
        with wave.open(str(path), "rb") as wf:
            sr   = wf.getframerate()
            ch   = wf.getnchannels()
            bits = wf.getsampwidth() * 8
            dur  = wf.getnframes() / sr
            info.update({"sr": sr, "ch": ch, "bits": bits, "dur": dur})
            issues = []
            if sr   != REQUIRED_SR:   issues.append(f"sample rate {sr} → need {REQUIRED_SR}")
            if ch   != REQUIRED_CH:   issues.append(f"{ch} channels → need mono")
            if bits != REQUIRED_BITS: issues.append(f"{bits}-bit → need {REQUIRED_BITS}-bit")
            if dur  < MIN_CLIP_SECS:  issues.append(f"too short ({dur:.2f}s)")
            if dur  > MAX_CLIP_SECS:  issues.append(f"too long ({dur:.2f}s)")
            info["issues"] = issues
            info["ok"] = len(issues) == 0
            if verbose:
                status = "OK" if info["ok"] else "NEEDS CONVERSION"
                print(f"      sample file: {path.name} — {sr}Hz, {ch}ch, {bits}bit, {dur:.2f}s [{status}]")
                for issue in issues:
                    print(f"        ⚠ {issue}")
    except Exception as e:
        info["error"] = str(e)
        if verbose:
            print(f"      ✖ Could not read {path.name}: {e}")
    return info


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — COLLECT AND VALIDATE AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

def collect_wavs(dirs: list[Path]) -> list[Path]:
    """Recursively collect all WAV files from a list of directories."""
    found = []
    for d in dirs:
        if d.exists():
            found.extend(sorted(d.glob("*.wav")))
            found.extend(sorted(d.glob("**/*.wav")))
    # Deduplicate (augmented/ might overlap with positive/)
    seen = set()
    unique = []
    for p in found:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def convert_wav(src: Path, dst: Path):
    """
    Convert a WAV to 16kHz / mono / int16 using scipy or soundfile.
    Falls back to a raw numpy resample if neither is available.
    """
    try:
        import soundfile as sf
        import scipy.signal as sig

        data, sr = sf.read(str(src), dtype="int16", always_2d=True)
        # Mix down to mono
        if data.shape[1] > 1:
            data = data.mean(axis=1).astype(np.int16)
        else:
            data = data[:, 0]
        # Resample if needed
        if sr != REQUIRED_SR:
            samples_new = int(len(data) * REQUIRED_SR / sr)
            data = sig.resample(data.astype(np.float32), samples_new).astype(np.int16)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(REQUIRED_SR)
            wf.writeframes(data.tobytes())

    except ImportError:
        # Minimal fallback — just fix channel count using raw wave module
        with wave.open(str(src), "rb") as wf_in:
            sr   = wf_in.getframerate()
            ch   = wf_in.getnchannels()
            raw  = wf_in.readframes(wf_in.getnframes())

        data = np.frombuffer(raw, dtype=np.int16)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)

        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as wf_out:
            wf_out.setnchannels(1)
            wf_out.setsampwidth(2)
            wf_out.setframerate(sr)   # keep original sr; resampling needs scipy
            wf_out.writeframes(data.tobytes())

        if sr != REQUIRED_SR:
            log.warning(
                f"  ⚠ {src.name}: sample rate is {sr}, not {REQUIRED_SR}. "
                f"Install scipy+soundfile for automatic resampling."
            )


def validate_and_prepare(
    positive_dirs: list[Path],
    negative_dirs: list[Path],
    val_split: float = 0.15,
) -> dict:
    """
    Validate all audio, auto-convert where possible, split into train/val.
    Returns a dict with keys: pos_train, pos_val, neg_train, neg_val.
    """
    print("  [STEP 1] Validating and preparing dataset...\n")

    pos_all = collect_wavs(positive_dirs)
    neg_all = collect_wavs(negative_dirs)

    # ── Friendly error with clear instructions ────────────────────────────────
    if not pos_all:
        _missing_data_error(positive_dirs, "positive")

    if not neg_all:
        log.warning(
            "  ⚠ No negative clips found. Training without negatives "
            "will produce a model with very high false-positive rate.\n"
            "  Add background/noise WAVs to: " +
            str(negative_dirs[0] if negative_dirs else DATASET_DIR / "negative")
        )

    log.info(f"  Found {len(pos_all)} positive clips, {len(neg_all)} negative clips")

    # ── Auto-convert any clips that need it ───────────────────────────────────
    converted_dir = DATASET_DIR / "_converted"
    converted_pos, converted_neg = [], []

    def _ensure_format(clips: list[Path], role: str) -> list[Path]:
        out = []
        for clip in clips:
            info = _check_wav_format(clip)
            if info.get("ok"):
                out.append(clip)
            else:
                dst = converted_dir / role / clip.name
                if not dst.exists():
                    log.info(f"  Converting {clip.name}: {info.get('issues', [])}")
                    convert_wav(clip, dst)
                out.append(dst)
        return out

    pos_all = _ensure_format(pos_all, "positive")
    neg_all = _ensure_format(neg_all, "negative")

    # ── Train / validation split ──────────────────────────────────────────────
    rng = np.random.default_rng(42)

    def split(clips):
        idx = rng.permutation(len(clips))
        cut = max(1, int(len(clips) * val_split))
        return [clips[i] for i in idx[cut:]], [clips[i] for i in idx[:cut]]

    pos_train, pos_val = split(pos_all)
    neg_train, neg_val = split(neg_all) if neg_all else ([], [])

    print(f"  Positive — train: {len(pos_train)}, val: {len(pos_val)}")
    print(f"  Negative — train: {len(neg_train)}, val: {len(neg_val)}\n")

    return {
        "pos_train": pos_train, "pos_val": pos_val,
        "neg_train": neg_train, "neg_val": neg_val,
    }


def _missing_data_error(dirs: list[Path], role: str):
    """Print a detailed, actionable error and exit cleanly."""
    print("\n" + "=" * 62)
    print(f"  ERROR: No {role} WAV clips found.")
    print("=" * 62)
    print(f"\n  Searched in:")
    for d in dirs:
        print(f"    {'EXISTS  ' if d.exists() else 'MISSING '} {d}")

    print(f"""
  HOW TO FIX:
  ───────────
  Your recordings from train_wake.py are in:
    C:\\Users\\hamza\\Desktop\\wake_training\\positive\\

  You need to copy them to:
    {DATASET_DIR / 'positive'}

  Run this in PowerShell:

    New-Item -ItemType Directory -Force "{DATASET_DIR / 'positive'}"
    Copy-Item "C:\\Users\\hamza\\Desktop\\wake_training\\positive\\*.wav" \\
              "{DATASET_DIR / 'positive'}\\"

  Then re-run:
    python -m wake.train --epochs 150

  OR pass the path explicitly:
    python -m wake.train --positive "C:\\Users\\hamza\\Desktop\\wake_training\\positive"
""")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def augment_dataset(
    pos_clips: list[Path],
    aug_dir: Path,
    target_count: int = 300,
) -> list[Path]:
    """
    Augment positive clips to reach target_count using:
    - Gaussian noise injection
    - Gain variation
    - Pitch shift (via sample rate trick)
    - Time stretch
    - Reverb simulation (simple convolution)

    Returns list of augmented WAV paths.
    """
    if len(pos_clips) >= target_count:
        log.info(f"  Already have {len(pos_clips)} clips ≥ target {target_count}. Skipping augmentation.")
        return pos_clips

    aug_dir.mkdir(parents=True, exist_ok=True)
    existing = list(aug_dir.glob("*.wav"))

    needed = target_count - len(pos_clips)
    log.info(f"  Augmenting: have {len(pos_clips)}, need {needed} more → target {target_count}")

    augmented = list(existing)  # reuse already-generated augments
    rng = np.random.default_rng(seed=99)
    idx = 0

    while len(augmented) < needed:
        src = pos_clips[idx % len(pos_clips)]
        idx += 1

        aug_name = f"aug_{idx:04d}_{src.stem}.wav"
        dst = aug_dir / aug_name
        if dst.exists():
            augmented.append(dst)
            continue

        try:
            # Load source
            with wave.open(str(src), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                sr  = wf.getframerate()
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

            # ── Choose a random augmentation ──────────────────────────────────
            aug_type = rng.integers(0, 6)

            if aug_type == 0:
                # Gaussian noise
                noise_level = rng.uniform(50, 800)
                data = data + rng.normal(0, noise_level, len(data))

            elif aug_type == 1:
                # Gain variation (±6 dB)
                gain = rng.uniform(0.5, 2.0)
                data = data * gain

            elif aug_type == 2:
                # Pitch shift via resampling trick (±15%)
                factor = rng.uniform(0.85, 1.15)
                n_new  = int(len(data) * factor)
                data   = np.interp(
                    np.linspace(0, len(data)-1, n_new),
                    np.arange(len(data)), data
                )
                # Trim or pad to original length
                if len(data) > len(np.frombuffer(raw, dtype=np.int16)):
                    data = data[:len(np.frombuffer(raw, dtype=np.int16))]
                else:
                    data = np.pad(data, (0, len(np.frombuffer(raw, dtype=np.int16)) - len(data)))

            elif aug_type == 3:
                # Time stretch (speed up / slow down) — resample then trim/pad
                factor = rng.uniform(0.85, 1.15)
                orig_len = len(data)
                new_len  = int(orig_len / factor)
                data     = np.interp(
                    np.linspace(0, orig_len-1, new_len),
                    np.arange(orig_len), data
                )
                if len(data) >= orig_len:
                    data = data[:orig_len]
                else:
                    data = np.pad(data, (0, orig_len - len(data)))

            elif aug_type == 4:
                # Simple reverb: mix with decayed delayed copy
                delay_samples = int(sr * rng.uniform(0.02, 0.08))
                decay = rng.uniform(0.2, 0.5)
                delayed = np.zeros_like(data)
                delayed[delay_samples:] = data[:-delay_samples] * decay
                data = data + delayed

            elif aug_type == 5:
                # Combined: noise + gain
                noise_level = rng.uniform(30, 400)
                gain        = rng.uniform(0.6, 1.8)
                data        = (data + rng.normal(0, noise_level, len(data))) * gain

            # ── Clip and save ──────────────────────────────────────────────────
            data = np.clip(data, -32768, 32767).astype(np.int16)
            with wave.open(str(dst), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(data.tobytes())

            augmented.append(dst)

        except Exception as e:
            log.warning(f"  Augmentation failed for {src.name}: {e}")

    log.info(f"  Augmentation complete: {len(augmented)} augmented clips")
    return pos_clips + augmented


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(clips: list[Path], label: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract log-mel spectrogram features from WAV clips.
    Returns (X, y) arrays suitable for training.

    Feature shape: (n_samples, n_frames, 32) — 32 mel bins, 3-second window.
    """
    try:
        import librosa
    except ImportError:
        log.error(
            "librosa not installed. Run: pip install librosa\n"
            "Or: pip install librosa soundfile scipy"
        )
        sys.exit(1)

    N_MELS   = 32
    HOP_LEN  = 160      # 10ms @ 16kHz
    WIN_LEN  = 400      # 25ms @ 16kHz
    N_FRAMES = 96       # ~1 second of context
    MAX_SECS = 3.0

    features, labels = [], []

    for clip in clips:
        try:
            y, sr = librosa.load(str(clip), sr=REQUIRED_SR, mono=True,
                                 duration=MAX_SECS)
            if len(y) < CHUNK_SAMPLES:
                continue

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS,
                hop_length=HOP_LEN, win_length=WIN_LEN,
            )
            log_mel = librosa.power_to_db(mel, ref=np.max).T  # (frames, mels)

            # Slide windows across the clip
            for start in range(0, max(1, len(log_mel) - N_FRAMES + 1), N_FRAMES // 2):
                window = log_mel[start: start + N_FRAMES]
                if len(window) < N_FRAMES:
                    # Pad if clip is shorter than N_FRAMES
                    window = np.pad(window, ((0, N_FRAMES - len(window)), (0, 0)),
                                    mode="constant", constant_values=-80)
                features.append(window)
                labels.append(label)

        except Exception as e:
            log.warning(f"  Feature extraction failed for {clip.name}: {e}")

    if not features:
        return np.empty((0, N_FRAMES, N_MELS)), np.empty(0, dtype=int)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(splits: dict, output_path: Path, epochs: int = 150):
    """
    Train the wake word model. Tries two backends in order:
    1. OpenWakeWord custom training API (preferred)
    2. scikit-learn + ONNX export (fallback, always available)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Try OpenWakeWord first ─────────────────────────────────────────────────
    success = _train_openwakeword(splits, output_path, epochs)
    if success:
        return

    # ── Fallback: sklearn MLP → ONNX ──────────────────────────────────────────
    log.warning("  OpenWakeWord training API unavailable. Using sklearn fallback.")
    _train_sklearn_onnx(splits, output_path, epochs)


def _train_openwakeword(splits: dict, output_path: Path, epochs: int) -> bool:
    """Try to train via the OpenWakeWord custom model API. Returns True on success."""
    try:
        # OWW 0.6+ exposes train_custom_model
        from openwakeword.custom import train_custom_model

        pos_dir = DATASET_DIR / "positive"
        aug_dir = DATASET_DIR / "augmented"
        neg_dir = DATASET_DIR / "negative"

        positive_refs = [str(pos_dir)]
        if aug_dir.exists():
            positive_refs.append(str(aug_dir))

        log.info("  Training via OpenWakeWord custom API...")
        train_custom_model(
            positive_reference_clips=positive_refs,
            negative_reference_clips=[str(neg_dir)] if neg_dir.exists() else [],
            output_dir=str(output_path.parent),
            model_name=output_path.stem,
            epochs=epochs,
            target_false_positive_rate=0.01,
        )

        if output_path.exists():
            log.info(f"  ✔ OpenWakeWord model saved: {output_path}")
            return True
        else:
            # OWW might have saved with a different name
            candidates = list(output_path.parent.glob("*.onnx"))
            if candidates:
                shutil.copy(candidates[0], output_path)
                log.info(f"  ✔ Model saved: {output_path}")
                return True

    except (ImportError, AttributeError) as e:
        log.warning(f"  OWW custom training API not found: {e}")
    except Exception as e:
        log.error(f"  OWW training failed: {e}")

    return False


def _train_sklearn_onnx(splits: dict, output_path: Path, epochs: int):
    """
    Fallback trainer: log-mel features → MLP → ONNX export.
    Works with any Python environment that has sklearn + onnxmltools.
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import sklearn
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    print("  [STEP 4] Extracting log-mel features...")
    all_pos = splits["pos_train"] + splits["pos_val"]
    all_neg = splits["neg_train"] + splits["neg_val"]

    X_pos, y_pos = extract_features(all_pos, label=1)
    X_neg, y_neg = extract_features(all_neg, label=0)

    if len(X_pos) == 0:
        log.error("  No features extracted from positive clips. Check audio format.")
        sys.exit(1)

    X = np.concatenate([X_pos, X_neg]) if len(X_neg) > 0 else X_pos
    y = np.concatenate([y_pos, y_neg]) if len(y_neg) > 0 else y_pos

    # Flatten (n_samples, frames, mels) → (n_samples, frames*mels)
    n_samples = len(X)
    X_flat = X.reshape(n_samples, -1)

    print(f"  Training MLP on {n_samples} samples ({len(X_pos)} positive, {len(X_neg)} negative)...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=epochs,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=True,
            random_state=42,
        )),
    ])
    pipeline.fit(X_flat, y)

    # ── Export to ONNX ────────────────────────────────────────────────────────
    _export_sklearn_onnx(pipeline, X_flat[:1], output_path)
    print(f"\n  ✔ Model trained and saved: {output_path}")
    _save_model_metadata(output_path, n_samples, len(X_pos), len(X_neg))


def _export_sklearn_onnx(pipeline, X_sample: np.ndarray, output_path: Path):
    """Export sklearn pipeline to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

        with open(str(output_path), "wb") as f:
            f.write(onnx_model.SerializeToString())

        log.info(f"  ONNX model exported: {output_path}")

    except ImportError:
        log.warning("  skl2onnx not installed. Saving as pickle fallback.")
        import pickle
        pkl_path = output_path.with_suffix(".pkl")
        with open(str(pkl_path), "wb") as f:
            pickle.dump(pipeline, f)
        log.info(f"  Saved as pickle: {pkl_path}")
        log.info("  Install skl2onnx for ONNX export: pip install skl2onnx")


def _save_model_metadata(output_path: Path, n_total: int, n_pos: int, n_neg: int):
    """Save training metadata alongside the model."""
    import json, datetime
    meta = {
        "model": output_path.name,
        "wake_word": "hey jarvis",
        "trained": datetime.datetime.now().isoformat(),
        "samples": {"total": n_total, "positive": n_pos, "negative": n_neg},
        "audio": {"sample_rate": REQUIRED_SR, "channels": REQUIRED_CH, "bits": REQUIRED_BITS},
    }
    meta_path = output_path.with_suffix(".json")
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Metadata saved: {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(output_path: Path, splits: dict):
    """Quick sanity check: run the saved model on validation clips and report accuracy."""
    if not output_path.exists():
        log.warning("  Model not found, skipping evaluation.")
        return

    print("\n  [STEP 5] Evaluating model on validation set...\n")

    pos_val = splits.get("pos_val", [])
    neg_val = splits.get("neg_val", [])

    if not pos_val and not neg_val:
        log.warning("  No validation clips available.")
        return

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path))
        input_name = sess.get_inputs()[0].name

        def _run(clips, expected_label):
            correct = 0
            for clip in clips:
                X, _ = extract_features([clip], label=expected_label)
                if len(X) == 0:
                    continue
                X_flat = X.reshape(len(X), -1).astype(np.float32)
                for sample in X_flat:
                    pred = sess.run(None, {input_name: sample[None]})[0][0]
                    if int(pred) == expected_label:
                        correct += 1
            return correct, len(clips)

        tp, total_pos = _run(pos_val, 1)
        tn, total_neg = _run(neg_val, 0)

        if total_pos > 0:
            print(f"  True positive rate  (sensitivity): {tp}/{total_pos} = {tp/total_pos*100:.1f}%")
        if total_neg > 0:
            print(f"  True negative rate  (specificity): {tn}/{total_neg} = {tn/total_neg*100:.1f}%")
        print()

    except Exception as e:
        log.warning(f"  Evaluation skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wake.train",
        description="Train a custom 'Hey JARVIS' wake word model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Standard run (uses wake/dataset/positive/ automatically)
  python -m wake.train --epochs 150

  # Point at your old wake_training folder
  python -m wake.train \\
    --positive "C:\\Users\\hamza\\Desktop\\wake_training\\positive" \\
    --negative "C:\\Users\\hamza\\Desktop\\wake_training\\negative" \\
    --epochs 150

  # Augment up to 400 clips before training
  python -m wake.train --epochs 150 --augment --target-clips 400

  # Diagnose only (no training)
  python -m wake.train --diagnose
        """
    )
    p.add_argument(
        "--positive", nargs="+", type=Path,
        help="Path(s) to positive 'Hey JARVIS' WAV clips. "
             f"Default: {[str(d) for d in DEFAULT_POSITIVE]}",
    )
    p.add_argument(
        "--negative", nargs="+", type=Path,
        help="Path(s) to negative/background WAV clips. "
             f"Default: {[str(d) for d in DEFAULT_NEGATIVE]}",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output ONNX model path. Default: {DEFAULT_OUTPUT}",
    )
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--val-split",     type=float, default=0.15)
    p.add_argument("--augment",       action="store_true",
                   help="Augment positive clips to reach --target-clips")
    p.add_argument("--target-clips",  type=int,   default=300,
                   help="Target clip count after augmentation (default: 300)")
    p.add_argument("--diagnose",      action="store_true",
                   help="Print diagnostics and exit without training")
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip post-training evaluation")
    return p


def run_training_pipeline(args):
    print_banner()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    # CRITICAL: use the script's own directory as the anchor, not cwd.
    positive_dirs = args.positive or DEFAULT_POSITIVE
    negative_dirs = args.negative or DEFAULT_NEGATIVE
    output_path   = args.output

    # Convert to Path and resolve relative paths against WAKE_DIR (not cwd)
    positive_dirs = [
        Path(p) if Path(p).is_absolute() else (WAKE_DIR / p).resolve()
        for p in positive_dirs
    ]
    negative_dirs = [
        Path(p) if Path(p).is_absolute() else (WAKE_DIR / p).resolve()
        for p in negative_dirs
    ]

    diagnose_paths(positive_dirs, negative_dirs)

    if args.diagnose:
        print("  [--diagnose] Exiting without training.")
        return

    # ── Validate + prepare ────────────────────────────────────────────────────
    splits = validate_and_prepare(positive_dirs, negative_dirs, val_split=args.val_split)

    # ── Augmentation ──────────────────────────────────────────────────────────
    if args.augment:
        aug_dir = DATASET_DIR / "augmented"
        splits["pos_train"] = augment_dataset(
            splits["pos_train"], aug_dir, target_count=args.target_clips
        )
        log.info(f"  Post-augmentation train set: {len(splits['pos_train'])} positive clips")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"  [STEP 4] Training ({args.epochs} epochs) → {output_path}\n")
    train_model(splits, output_path, epochs=args.epochs)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if not args.skip_eval:
        evaluate_model(output_path, splits)

    # ── Final instructions ────────────────────────────────────────────────────
    print("=" * 62)
    print("  TRAINING COMPLETE")
    print("=" * 62)
    print(f"\n  Model: {output_path}")
    print(f"\n  To use this model, update voice.py:")
    print(f'    wakeword_models=["{output_path}"]')
    print(f"\n  Or run the JARVIS detector:")
    print(f"    python -m wake.detector --model \"{output_path}\"")
    print()


if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()
    run_training_pipeline(args)
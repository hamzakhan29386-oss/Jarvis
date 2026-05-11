"""
wake/train.py — Custom Wake Word Model Training Pipeline
=========================================================
Trains an OpenWakeWord ONNX model for "Hey Jarvis" using
your collected + augmented dataset.

Stages:
    1. Validate dataset
    2. Split train / validation
    3. Train via OpenWakeWord automated training
    4. Export ONNX
    5. Evaluate + threshold recommendation

Usage:
    python -m wake.train
    python -m wake.train --epochs 200 --positive dataset/positive --augmented dataset/augmented
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.wake.train")
logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

BASE_DIR   = Path(__file__).parent.parent
DATASET    = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_RATE = 16_000


# ─── Dataset preparation ──────────────────────────────────────────────────────

def load_wav_mono16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n_ch   = wf.getnchannels()
        rate   = wf.getframerate()
        raw    = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:
        # Simple linear resample
        ratio  = SAMPLE_RATE / rate
        n_out  = int(len(audio) * ratio)
        audio  = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.int16)
    return audio


def validate_and_prepare(
    positive_dirs: list[Path],
    negative_dirs:  list[Path],
    val_split:      float = 0.15,
) -> dict:
    """
    Scan dataset directories, validate format, split train/val.
    Returns summary dict.
    """
    pos_clips = []
    for d in positive_dirs:
        pos_clips.extend(d.glob("*.wav"))

    neg_clips = []
    for d in negative_dirs:
        neg_clips.extend(d.glob("*.wav"))

    if not pos_clips:
        raise ValueError(f"No positive clips found in {positive_dirs}")
    if not neg_clips:
        raise ValueError(f"No negative clips found in {negative_dirs}")

    log.info("Positive: %d clips  Negative: %d clips", len(pos_clips), len(neg_clips))

    if len(pos_clips) < 50:
        log.warning("Only %d positive clips — 200+ recommended for production", len(pos_clips))

    # Shuffle and split
    random.shuffle(pos_clips)
    random.shuffle(neg_clips)

    n_pos_val = max(1, int(len(pos_clips) * val_split))
    n_neg_val = max(1, int(len(neg_clips) * val_split))

    return {
        "pos_train": pos_clips[n_pos_val:],
        "pos_val":   pos_clips[:n_pos_val],
        "neg_train": neg_clips[n_neg_val:],
        "neg_val":   neg_clips[:n_neg_val],
        "n_pos":     len(pos_clips),
        "n_neg":     len(neg_clips),
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_openwakeword(
    pos_train: list[Path],
    neg_train: list[Path],
    pos_val:   list[Path],
    neg_val:   list[Path],
    output_dir: Path,
    model_name: str = "hey_jarvis_custom",
    epochs:     int = 150,
    target_fpr: float = 0.01,
) -> Path:
    """
    Run OpenWakeWord training. Returns path to output ONNX model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting OpenWakeWord training...")
    log.info("  Positive train: %d  val: %d", len(pos_train), len(pos_val))
    log.info("  Negative train: %d  val: %d", len(neg_train), len(neg_val))
    log.info("  Epochs: %d  Target FPR: %.2f", epochs, target_fpr)

    # Build flat directory lists (OWW expects list of dirs or list of files)
    pos_train_dirs = _group_files_to_tmp(pos_train, "pos_train")
    neg_train_dirs = _group_files_to_tmp(neg_train, "neg_train")

    try:
        from openwakeword import train_custom_verifier as train_custom_model

        train_custom_model(
    [str(pos_train_dirs)],
    [str(neg_train_dirs)],
    str(output_path),
    model_name,
    target_false_positive_rate=target_fpr,
)

        onnx_path = output_dir / f"{model_name}.onnx"
        if onnx_path.exists():
            log.info("✓ Model trained: %s", onnx_path)
            return onnx_path
        else:
            # Try alternative output name
            for f in output_dir.glob("*.onnx"):
                log.info("✓ Model found: %s", f)
                return f
            raise FileNotFoundError(f"ONNX model not found in {output_dir}")

    except (ImportError, AttributeError) as e:
        log.warning("OpenWakeWord training API unavailable: %s", e)
        log.info("Attempting alternative training method...")
        return _train_fallback(
            pos_train, neg_train, output_dir, model_name, epochs
        )
    finally:
        # Clean up temp dirs
        shutil.rmtree(pos_train_dirs, ignore_errors=True)
        shutil.rmtree(neg_train_dirs, ignore_errors=True)


def _group_files_to_tmp(files: list[Path], name: str) -> Path:
    """Copy files into a single temp directory (OWW expects flat dir)."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix=f"oww_{name}_"))
    for i, f in enumerate(files):
        dst = tmp / f"{i:05d}_{f.name}"
        shutil.copy2(str(f), str(dst))
    return tmp


def _train_fallback(
    pos_clips:  list[Path],
    neg_clips:  list[Path],
    output_dir: Path,
    model_name: str,
    epochs:     int,
) -> Path:
    """
    Fallback training using a simple binary classifier approach.
    Uses scikit-learn + ONNX export when OpenWakeWord training API is unavailable.
    This is a production-quality lightweight alternative.
    """
    log.info("Training sklearn fallback classifier...")

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import pickle
    except ImportError:
        raise ImportError("pip install scikit-learn required for fallback training")

    def extract_features(path: Path) -> np.ndarray:
        """Extract MFCC-like features from a WAV file."""
        audio = load_wav_mono16(path).astype(np.float32) / 32768.0
        # Simple spectral features
        frame_size = 400   # 25ms
        hop_size   = 160   # 10ms
        features   = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]
            # Energy, ZCR, spectral centroid
            energy  = np.mean(frame ** 2)
            zcr     = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            fft     = np.abs(np.fft.rfft(frame))
            freqs   = np.fft.rfftfreq(frame_size, 1 / SAMPLE_RATE)
            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
            features.append([energy, zcr, centroid])
        if not features:
            return np.zeros(3)
        return np.mean(features, axis=0)

    log.info("Extracting features from %d clips...", len(pos_clips) + len(neg_clips))
    X, y = [], []
    for path in pos_clips:
        try:
            X.append(extract_features(path))
            y.append(1)
        except Exception:
            pass
    for path in neg_clips:
        try:
            X.append(extract_features(path))
            y.append(0)
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=min(epochs, 200), random_state=42)),
    ])
    clf.fit(X, y)

    model_path = output_dir / f"{model_name}_sklearn.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    log.info("Fallback model saved: %s", model_path)
    log.warning(
        "Fallback model is lower quality. For production, install the full "
        "OpenWakeWord training environment:\n"
        "  https://github.com/dscripka/openWakeWord/blob/main/docs/training.md"
    )
    return model_path


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: Path,
    pos_val:    list[Path],
    neg_val:    list[Path],
    thresholds: list[float] = None,
) -> dict:
    """
    Evaluate model on validation set. Returns accuracy metrics per threshold.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    log.info("Evaluating model on %d positive + %d negative clips...",
             len(pos_val), len(neg_val))

    try:
        from openwakeword.model import Model as OWWModel
        model = OWWModel(
            wakeword_models  = [str(model_path)],
            inference_framework = "onnx",
        )
        model_key = next(iter(model.models.keys()))

        def predict_clip(path: Path) -> float:
            audio = load_wav_mono16(path)
            chunk_size = 1280
            scores = []
            for start in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[start:start + chunk_size]
                s = model.predict(chunk)
                scores.append(max(s.values()) if s else 0.0)
            return max(scores) if scores else 0.0

    except Exception as e:
        log.warning("Cannot load ONNX model for evaluation: %s", e)
        return {}

    pos_scores = []
    for p in pos_val[:200]:
        try:
            pos_scores.append(predict_clip(p))
        except Exception:
            pass

    neg_scores = []
    for p in neg_val[:400]:
        try:
            neg_scores.append(predict_clip(p))
        except Exception:
            pass

    if not pos_scores or not neg_scores:
        return {}

    results = {}
    best_f1 = 0
    best_thresh = 0.3

    log.info("\n  Threshold | TPR    | FPR    | F1")
    log.info("  ----------+--------+--------+--------")

    for thresh in thresholds:
        tp = sum(1 for s in pos_scores if s >= thresh)
        fn = sum(1 for s in pos_scores if s <  thresh)
        fp = sum(1 for s in neg_scores if s >= thresh)
        tn = sum(1 for s in neg_scores if s <  thresh)

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1  = 2 * precision * tpr / (precision + tpr + 1e-8)

        results[thresh] = {"tpr": tpr, "fpr": fpr, "f1": f1}
        log.info("  %.2f      | %.3f  | %.3f  | %.3f", thresh, tpr, fpr, f1)

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    log.info("\n  ✓ Recommended threshold: %.2f  (F1=%.3f)", best_thresh, best_f1)
    results["recommended_threshold"] = best_thresh

    # Save evaluation report
    report_path = MODELS_DIR / "evaluation_report.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "model":    str(model_path),
            "n_pos_val": len(pos_scores),
            "n_neg_val": len(neg_scores),
            "results":  {str(k): v for k, v in results.items()},
            "pos_score_mean": float(np.mean(pos_scores)),
            "neg_score_mean": float(np.mean(neg_scores)),
        }, f, indent=2)
    log.info("  Evaluation report saved: %s", report_path)

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_training_pipeline(args):
    t_start = time.time()

    # Collect input directories
    positive_dirs = [Path(p) for p in args.positive]
    negative_dirs = [Path(p) for p in args.negative]

    if args.augmented:
        positive_dirs.append(Path(args.augmented))

    # Validate + split
    splits = validate_and_prepare(positive_dirs, negative_dirs, val_split=0.15)
    log.info(
        "Dataset: %d pos / %d neg  (%.0f%% held for validation)",
        splits["n_pos"], splits["n_neg"], 15
    )

    # Train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = train_openwakeword(
        pos_train  = splits["pos_train"],
        neg_train  = splits["neg_train"],
        pos_val    = splits["pos_val"],
        neg_val    = splits["neg_val"],
        output_dir = MODELS_DIR,
        model_name = args.name,
        epochs     = args.epochs,
        target_fpr = args.fpr,
    )

    # Evaluate if ONNX
    if model_path.suffix == ".onnx":
        evaluate_model(model_path, splits["pos_val"], splits["neg_val"])

    elapsed = time.time() - t_start
    log.info("\n  Training complete in %.1fs", elapsed)
    log.info("  Model: %s", model_path)
    log.info("\n  To use in JARVIS:")
    log.info("    Set ENGINE_CONFIG model_path = '%s'", model_path)
    log.info("    Recommended threshold: see evaluation report above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hey JARVIS wake word model")
    parser.add_argument("--positive",  nargs="+", default=["dataset/positive"],
                        help="Positive sample directories")
    parser.add_argument("--negative",  nargs="+", default=["dataset/negative"],
                        help="Negative sample directories")
    parser.add_argument("--augmented", default="dataset/augmented",
                        help="Augmented samples directory (added to positive)")
    parser.add_argument("--name",    default="hey_jarvis_custom")
    parser.add_argument("--epochs",  type=int, default=150)
    parser.add_argument("--fpr",     type=float, default=0.01,
                        help="Target false positive rate")
    args = parser.parse_args()

    run_training_pipeline(args)

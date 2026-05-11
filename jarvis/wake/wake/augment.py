"""
wake/augment.py — Audio Augmentation Pipeline
==============================================
Transforms raw recordings into a robust training dataset.
Implements the same augmentation strategies used by Alexa/Google.

Augmentations:
    - Background noise mixing (fan, TV, music, café)
    - Room impulse response / reverb simulation
    - Pitch shifting (±2 semitones)
    - Gain variation (±12 dB)
    - Time stretching (0.85x–1.15x)
    - Band-pass filtering (telephone effect)
    - Gaussian noise injection
    - Echo simulation
    - Microphone distance simulation (lowpass + gain drop)

Usage:
    python -m wake.augment --input dataset/positive --output dataset/augmented --factor 5
"""

from __future__ import annotations

import argparse
import logging
import random
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.wake.augment")

SAMPLE_RATE = 16_000


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != SAMPLE_RATE:
        audio = _resample(audio, rate, SAMPLE_RATE)
    return audio


def save_wav(path: Path, audio: np.ndarray):
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    ratio = dst / src
    n_out = int(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)


# ─── Individual augmentations ─────────────────────────────────────────────────

def add_gaussian_noise(audio: np.ndarray, snr_db: float = None) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (dB)."""
    snr_db = snr_db if snr_db is not None else random.uniform(10, 35)
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def change_gain(audio: np.ndarray, db: float = None) -> np.ndarray:
    """Randomly scale gain (simulates mic distance / volume variation)."""
    db = db if db is not None else random.uniform(-10, 6)
    factor = 10 ** (db / 20)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def pitch_shift(audio: np.ndarray, semitones: float = None) -> np.ndarray:
    """Shift pitch by ±N semitones via resampling trick."""
    semitones = semitones if semitones is not None else random.uniform(-2, 2)
    factor = 2 ** (semitones / 12)
    # Stretch then resample back to original length
    n_stretched = int(len(audio) / factor)
    stretched   = np.interp(
        np.linspace(0, len(audio) - 1, n_stretched),
        np.arange(len(audio)),
        audio
    )
    return np.interp(
        np.linspace(0, len(stretched) - 1, len(audio)),
        np.arange(len(stretched)),
        stretched
    ).astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float = None) -> np.ndarray:
    """Stretch/compress time without changing pitch (naive resampling)."""
    rate  = rate if rate is not None else random.uniform(0.88, 1.15)
    n_out = int(len(audio) * rate)
    stretched = np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio
    ).astype(np.float32)
    # Trim or pad back to original length
    if len(stretched) >= len(audio):
        return stretched[:len(audio)]
    return np.pad(stretched, (0, len(audio) - len(stretched)))


def add_echo(audio: np.ndarray, delay_ms: float = None, decay: float = None) -> np.ndarray:
    """Simulate simple room echo."""
    delay_ms = delay_ms if delay_ms is not None else random.uniform(50, 250)
    decay    = decay    if decay    is not None else random.uniform(0.2, 0.5)
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    out = audio.copy()
    if delay_samples < len(out):
        out[delay_samples:] += decay * audio[:-delay_samples]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def simulate_reverb(audio: np.ndarray, room_size: float = None) -> np.ndarray:
    """
    Cheap reverb: convolve with exponentially-decaying impulse response.
    room_size: 0.0 (small) → 1.0 (large hall)
    """
    room_size = room_size if room_size is not None else random.uniform(0.1, 0.6)
    ir_len    = int(SAMPLE_RATE * room_size * 0.5)   # up to 300ms
    ir        = np.random.randn(ir_len).astype(np.float32)
    decay     = np.exp(-np.linspace(0, 6, ir_len))
    ir        *= decay
    ir        /= (np.max(np.abs(ir)) + 1e-8)
    reverbed  = np.convolve(audio, ir, mode="full")[:len(audio)]
    # Blend dry + wet
    wet = random.uniform(0.15, 0.4)
    return np.clip((1 - wet) * audio + wet * reverbed, -1.0, 1.0).astype(np.float32)


def telephone_effect(audio: np.ndarray) -> np.ndarray:
    """Band-pass filter to simulate low-quality microphone / phone call."""
    try:
        from scipy.signal import butter, lfilter
        # Telephone band: 300Hz – 3400Hz
        lo  = 300  / (SAMPLE_RATE / 2)
        hi  = 3400 / (SAMPLE_RATE / 2)
        b, a = butter(4, [lo, hi], btype="band")
        return lfilter(b, a, audio).astype(np.float32)
    except ImportError:
        return audio   # gracefully skip


def mix_background_noise(
    audio:      np.ndarray,
    noise_dir:  Path,
    snr_db:     float = None,
) -> np.ndarray:
    """
    Mix a random background noise clip from noise_dir at a target SNR.
    Falls back to Gaussian noise if no noise files are available.
    """
    snr_db = snr_db if snr_db is not None else random.uniform(8, 25)
    noise_clips = list(noise_dir.glob("*.wav")) if noise_dir.exists() else []

    if noise_clips:
        noise_path = random.choice(noise_clips)
        noise = load_wav(noise_path)
        # Loop noise to match signal length
        if len(noise) < len(audio):
            repeats = (len(audio) // len(noise)) + 1
            noise   = np.tile(noise, repeats)
        noise = noise[:len(audio)]
    else:
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.01

    # Scale noise to achieve target SNR
    signal_rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    noise_rms  = np.sqrt(np.mean(noise  ** 2)) + 1e-10
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)

    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def simulate_distance(audio: np.ndarray, meters: float = None) -> np.ndarray:
    """
    Simulate microphone distance: gain drops with 1/r² + soft lowpass.
    meters: 0.3 (close) → 5.0 (far)
    """
    meters = meters if meters is not None else random.uniform(0.5, 4.0)
    # Gain drop: -6dB per doubling of distance (inverse square)
    reference_m = 0.3
    gain_db = -20 * np.log10(max(meters / reference_m, 1.0))
    audio   = change_gain(audio, db=gain_db)
    # High-frequency roll-off (air absorption + distance)
    try:
        from scipy.signal import butter, lfilter
        cutoff = max(1000, 8000 - int(meters * 1200))
        b, a   = butter(2, cutoff / (SAMPLE_RATE / 2), btype="low")
        audio  = lfilter(b, a, audio).astype(np.float32)
    except ImportError:
        pass
    return audio


# ─── Augmentation pipeline ────────────────────────────────────────────────────

# Named augmentation strategies with weights (higher = more frequent)
AUGMENTATIONS = [
    ("gaussian_noise",    add_gaussian_noise,   3),
    ("gain_change",       change_gain,           3),
    ("pitch_shift",       pitch_shift,           2),
    ("time_stretch",      time_stretch,          2),
    ("echo",              add_echo,              2),
    ("reverb",            simulate_reverb,       2),
    ("distance",          simulate_distance,     2),
    ("telephone",         telephone_effect,      1),
]

# Weighted random selection
_AUG_CHOICES = [(name, fn) for name, fn, w in AUGMENTATIONS for _ in range(w)]


def augment_clip(
    audio:     np.ndarray,
    noise_dir: Path,
    n_augments: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """
    Apply 1–3 random augmentations to one clip.
    Returns (augmented_audio, list_of_augmentation_names).
    """
    applied = []
    aug_audio = audio.copy()

    # Always mix background noise (most important for real-world robustness)
    aug_audio = mix_background_noise(aug_audio, noise_dir)
    applied.append("bg_noise")

    # Apply additional random augmentations
    selected = random.sample(_AUG_CHOICES, min(n_augments, len(_AUG_CHOICES)))
    for name, fn in selected:
        if name not in applied:
            try:
                aug_audio = fn(aug_audio)
                applied.append(name)
            except Exception as e:
                log.debug("[Augment] %s failed: %s", name, e)

    return aug_audio, applied


def run_augmentation_pipeline(
    input_dir:  Path,
    output_dir: Path,
    noise_dir:  Path,
    factor:     int  = 5,
    max_clips:  int  = 5000,
):
    """
    Augment all WAV files in input_dir by `factor` times.
    Output goes to output_dir.

    Args:
        input_dir:  positive/ or negative/
        output_dir: augmented/
        noise_dir:  negative/ (used as background noise source)
        factor:     how many augmented variants per original clip
        max_clips:  safety cap on total output
    """
    clips = list(input_dir.glob("*.wav"))
    if not clips:
        print(f"  [Augment] No WAV files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    prefix    = input_dir.name  # "positive" or "negative"

    print(f"\n  [Augment] {len(clips)} clips × {factor} = up to {len(clips)*factor} variants")
    print(f"  [Augment] Output → {output_dir}\n")

    for i, clip_path in enumerate(clips):
        if generated >= max_clips:
            break
        try:
            audio = load_wav(clip_path)
        except Exception as e:
            log.warning("[Augment] Could not load %s: %s", clip_path, e)
            continue

        for v in range(factor):
            if generated >= max_clips:
                break
            aug_audio, applied = augment_clip(audio, noise_dir)
            out_name = f"{prefix}_aug_{i+1:04d}_v{v+1}.wav"
            save_wav(output_dir / out_name, aug_audio)
            generated += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(clips):
            pct = (i + 1) / len(clips) * 100
            print(f"  [{i+1:4d}/{len(clips)}] {pct:.0f}%  ({generated} clips generated)")

    print(f"\n  [Augment] Done. Generated {generated} augmented clips.")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment wake word dataset")
    parser.add_argument("--input",  default="dataset/positive", help="Input directory")
    parser.add_argument("--noise",  default="dataset/negative", help="Noise source directory")
    parser.add_argument("--output", default="dataset/augmented", help="Output directory")
    parser.add_argument("--factor", type=int, default=5,  help="Augmentations per clip")
    parser.add_argument("--max",    type=int, default=5000, help="Max output clips")
    args = parser.parse_args()

    BASE = Path(__file__).parent.parent
    run_augmentation_pipeline(
        input_dir  = BASE / args.input,
        output_dir = BASE / args.output,
        noise_dir  = BASE / args.noise,
        factor     = args.factor,
        max_clips  = args.max,
    )

"""
train_wake.py — Record your voice and train a custom "Hey JARVIS" wake word
============================================================================
Step 1: Records you saying "Hey JARVIS" 20 times
Step 2: Records background noise for negative examples
Step 3: Trains a personal openwakeword model tuned to YOUR voice
Step 4: Saves model to jarvis_custom.onnx

Run:  python train_wake.py
Then update voice.py to use the new model.
"""

import os
import time
import wave
import numpy as np
import sounddevice as sd

RATE = 16000
DEVICE = 1          # Intel mic array
CHANNELS = 4        # device has 4 channels
RECORD_SECONDS = 2  # each clip length
N_POSITIVE = 20     # "Hey JARVIS" recordings
N_NEGATIVE = 10     # background noise recordings
OUT_DIR = r"C:\Users\hamza\Desktop\wake_training"

os.makedirs(f"{OUT_DIR}/positive", exist_ok=True)
os.makedirs(f"{OUT_DIR}/negative", exist_ok=True)

def record_clip(duration=2.0, label=""):
    """Record a single audio clip, return mono int16 numpy array."""
    frames = []
    done = False

    def cb(indata, f, t, status):
        frames.append(indata[:, 0].copy())  # channel 0 only

    with sd.InputStream(device=DEVICE, samplerate=RATE, channels=CHANNELS,
                        dtype="int16", blocksize=1024, callback=cb):
        time.sleep(duration)

    return np.concatenate(frames)

def save_wav(path, data):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(data.tobytes())

print("\n" + "="*55)
print("  JARVIS Custom Wake Word Trainer")
print("="*55)
print("\n  Step 1: Recording positive examples (Hey JARVIS)")
print("  Say 'Hey JARVIS' clearly after each beep.")
print("  Vary your tone slightly each time.\n")

input("  Press Enter when ready...")

for i in range(N_POSITIVE):
    print(f"\n  [{i+1}/{N_POSITIVE}] Get ready...", end="", flush=True)
    time.sleep(0.8)
    print(" SPEAK NOW → ", end="", flush=True)
    data = record_clip(duration=2.0)
    path = f"{OUT_DIR}/positive/hey_jarvis_{i+1:02d}.wav"
    save_wav(path, data)
    print(f"saved ({len(data)} samples)")
    time.sleep(0.3)

print("\n\n  Step 2: Recording background noise (stay SILENT)")
print("  This helps the model reject false positives.\n")
input("  Press Enter when ready, then stay quiet...")

for i in range(N_NEGATIVE):
    print(f"\n  [{i+1}/{N_NEGATIVE}] Recording silence...", end="", flush=True)
    data = record_clip(duration=2.0)
    path = f"{OUT_DIR}/negative/noise_{i+1:02d}.wav"
    save_wav(path, data)
    print("saved")

print("\n\n  Step 3: Training custom model...")
print("  This may take 1-2 minutes.\n")

try:
    from openwakeword.custom import train_custom_model
    train_custom_model(
        positive_reference_clips=[f"{OUT_DIR}/positive"],
        negative_reference_clips=[f"{OUT_DIR}/negative"],
        output_dir=".",
        model_name="hey_jarvis_custom",
        epochs=100,
        target_false_positive_rate=0.01,
    )
    print("\n  Done! Model saved as: hey_jarvis_custom.onnx")
    print("\n  Now update voice.py:")
    print('  Change: wakeword_models=["hey_jarvis_v0.1"]')
    print('  To:     wakeword_models=["hey_jarvis_custom.onnx"]')

except AttributeError:
    # Older openwakeword API
    try:
        import openwakeword
        print("  Trying alternative training API...")
        from openwakeword import train
        train.train_model(
            positive_clips=f"{OUT_DIR}/positive",
            negative_clips=f"{OUT_DIR}/negative",
            output_path="hey_jarvis_custom.onnx",
            model_name="hey_jarvis_custom",
        )
        print("\n  Done! Model saved as: hey_jarvis_custom.onnx")
    except Exception as e:
        print(f"\n  Training API not available in this version: {e}")
        print("\n  Your recordings are saved in wake_training/")
        print("  Use the openwakeword training notebook instead:")
        print("  https://github.com/dscripka/openWakeWord/blob/main/notebooks/automated_model_training.ipynb")
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

model = Model(wakeword_models=["hey_jarvis_v0.1"], inference_framework="onnx")
print("Loaded. Say Hey JARVIS repeatedly...")
buf_ready = False
latest_buf = None

def cb(indata, frames, t, status):
    global latest_buf, buf_ready
    latest_buf = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
    buf_ready = True

with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280, callback=cb):
    while True:
        if buf_ready and latest_buf is not None:
            buf_ready = False
            p = model.predict(latest_buf)
            for k, s in p.items():
                if "hey_jarvis" in k:
                    bar = "X" * int(s * 40)
                    tag = "DETECTED" if s > 0.3 else ""
                    print(f"  {s:.4f} |{bar:<40}| {tag}", end="\r")
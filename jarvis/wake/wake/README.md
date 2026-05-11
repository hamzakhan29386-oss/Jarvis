# 🎙️ JARVIS Production Wake Word System

A production-grade always-listening wake word engine for the JARVIS AI assistant.  
Comparable in reliability to **Alexa**, **Siri**, and **Google Assistant**.

---

## ⚡ Architecture

```
Microphone
    │
    ▼  [Audio Thread — non-blocking, lock-free queue]
RMS Gate  ──── below threshold? ──► skip (saves CPU)
    │
    ▼  [Processing Thread]
WebRTC VAD  ──── non-speech? ──► skip
    │
    ▼
RNNoise / noisereduce
    │
    ▼
OpenWakeWord  ──── rolling confidence window ────► avg score
    │
    ▼
Threshold check  ──── below threshold? ──► skip
    │
    ▼
Cooldown guard  ──── triggered too recently? ──► skip
    │
    ▼
Resemblyzer Speaker Verification  ──── not owner? ──► reject
    │
    ▼  [Confirmed wake]
★ DETECTED ★
    │
    ▼
Faster-Whisper STT (command capture)
    │
    ▼
SSE event → Frontend HUD
    │
    ▼
AI Brain (brain.py → think_stream)
```

---

## 📁 File Structure

```
jarvis_wake/
├── wake/
│   ├── engine.py           ← Core production detector
│   ├── service.py          ← Flask/SSE integration
│   ├── collect_dataset.py  ← Dataset recording tool
│   ├── augment.py          ← Data augmentation pipeline
│   └── train.py            ← Model training pipeline
├── tools/
│   ├── mic_diagnostics.py  ← Audio setup validator
│   ├── threshold_tuner.py  ← Live confidence visualizer
│   ├── false_positive_test.py ← FPR benchmark
│   └── speaker_enroll.py   ← Owner voice enrollment
├── dataset/
│   ├── positive/           ← "Hey JARVIS" recordings
│   ├── negative/           ← Background noise
│   ├── augmented/          ← Augmented positive samples
│   └── validation/         ← Held-out test set
├── models/
│   ├── hey_jarvis_custom.onnx  ← Trained model (generated)
│   └── owner_voice.npy         ← Speaker profile (generated)
├── logs/
└── requirements.txt
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Diagnose your microphone

```bash
python tools/mic_diagnostics.py
```

Fix any issues before proceeding.

### Step 3 — Test with built-in model (no training required)

```bash
python tools/threshold_tuner.py
```

Say "Hey JARVIS" repeatedly and watch the confidence scores.  
Note the peak scores and background noise level.

### Step 4 — Record your dataset

```bash
# Record 50+ positive samples (say "Hey JARVIS" each time)
python -m wake.collect_dataset positive --clips 50

# Record background noise (stay quiet or let ambient sounds play)
python -m wake.collect_dataset negative --clips 60

# Check dataset quality
python -m wake.collect_dataset validate
```

### Step 5 — Augment the dataset

```bash
python -m wake.augment --factor 5
```

This generates 5 augmented variants per clip (noise, echo, reverb, pitch, etc).

### Step 6 — Train your custom model

```bash
python -m wake.train --epochs 150
```

Training takes 5–20 minutes depending on dataset size.

### Step 7 — Evaluate and tune threshold

```bash
# Benchmark false positives
python tools/false_positive_test.py --threshold 0.5 --window 3

# Live threshold tuning with your custom model
python tools/threshold_tuner.py --model models/hey_jarvis_custom.onnx
```

### Step 8 — Enroll your voice (optional but recommended)

```bash
python tools/speaker_enroll.py enroll
python tools/speaker_enroll.py test
```

### Step 9 — Integrate with JARVIS server

In `server.py`, change:
```python
# Old:
from voice import get_wake_service

# New:
from wake.service import get_wake_service
```

That's it. All existing SSE endpoints (`/voice/wake-stream`, `/voice/wake-enable`, etc.)  
work identically with the production service.

---

## ⚙️ Configuration

### EngineConfig parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.35` | Detection confidence threshold. Higher = fewer false positives. |
| `rolling_window` | `3` | Average last N chunks. Higher = more stable, slightly more latency. |
| `cooldown_secs` | `2.5` | Seconds to ignore after each detection (prevents repeats). |
| `rms_gate` | `80` | Skip processing if audio RMS below this (saves CPU). |
| `model_path` | `None` | Path to custom ONNX model. None = built-in hey_jarvis_v0.1. |
| `debug_scores` | `False` | Print live scores to stdout (for tuning). |

### Threshold guide

| Threshold | Behavior |
|-----------|----------|
| `0.25–0.35` | Very sensitive. May have false positives. Good for tuning/debugging. |
| `0.40–0.50` | Balanced. Start here for production with custom model. |
| `0.55–0.70` | Conservative. Very few false positives. May miss quieter speech. |
| `0.75+` | Very strict. Only triggers on clear, close-range speech. |

### Rolling window guide

| Window | Behavior |
|--------|----------|
| `1` | Instant response. Higher false positive risk. Use for debugging. |
| `2` | Good balance. First step up from 1 during testing. |
| `3` | Recommended production setting. Smooth, reliable. |
| `5` | Very stable. Adds ~400ms latency. Good for noisy environments. |

---

## 🎯 Performance Targets

| Metric | Target | How to achieve |
|--------|--------|----------------|
| Wake latency | < 300ms | Rolling window = 1–2, threshold = 0.40 |
| False positive rate | < 0.1/min | Threshold ≥ 0.50, window ≥ 3, speaker verification |
| CPU usage | < 5% | RMS gate, VAD pre-filter |
| Detection rate | > 95% | 200+ training clips, augmentation, threshold = 0.40 |

---

## 🐛 Common Failure Modes & Fixes

### "Model failed to load"
```bash
pip install openwakeword
# If that fails:
pip install onnxruntime openwakeword
```

### "Stream failed — invalid device"
```bash
python tools/mic_diagnostics.py  # shows all devices
python tools/threshold_tuner.py --device 2  # specify device index
```

### "Score always 0.0000"
- Wrong sample rate. Check mic_diagnostics output.
- Model not matching key. Enable `debug_scores=True` to see raw dict.
- Stereo audio. The engine forces mono — check that `channels=1` in stream.

### High false positive rate
1. Raise threshold: `threshold=0.55`
2. Increase window: `rolling_window=4`
3. Enable speaker verification: `python tools/speaker_enroll.py enroll`
4. Collect more negative samples and retrain

### Detection misses (wake word not recognized)
1. Lower threshold: `threshold=0.30`
2. Decrease window: `rolling_window=1` for debugging
3. Collect more positive samples (try to get 200+ clips)
4. Vary recording conditions during dataset collection

### High CPU usage
1. Increase `rms_gate` to `200–500` to skip quiet audio
2. VAD is already filtering — ensure webrtcvad is installed: `pip install webrtcvad`
3. Use `tiny.en` Whisper model (default)

---

## 📊 Dataset Collection Tips (Alexa-level quality)

### For positive samples, record in:
- [ ] Your desk (normal use case)  
- [ ] Kitchen with background noise  
- [ ] Different rooms (bathroom echo, living room reverb)  
- [ ] With TV playing in background  
- [ ] With music at moderate volume  
- [ ] From 1m, 2m, 3m away from mic  
- [ ] Morning voice (sleepy)  
- [ ] Urgent/excited tone  
- [ ] Whispering  
- [ ] After exercise (slightly breathless)  

### For negative samples, include:
- [ ] Complete silence  
- [ ] Fan / HVAC noise  
- [ ] TV speech (news, movies)  
- [ ] Music (various genres)  
- [ ] Random words and sentences  
- [ ] Other people's voices  
- [ ] Keyboard and mouse sounds  
- [ ] "Hey Google", "OK Alexa" (common false triggers)  
- [ ] Words similar to "Jarvis" (Ferris, Paris, Marcus)  

---

## 🔌 Server Integration (server.py diff)

```python
# ─── Old (in server.py) ─────────────────────────────────────────
from voice import get_wake_service

# ─── New ────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wake.service import get_wake_service

# Everything else in server.py stays IDENTICAL.
# /voice/wake-enable, /voice/wake-disable, /voice/wake-stream
# all work without any changes.
```

---

## 📦 Dependencies

```
openwakeword>=0.6.0        # wake word detection
faster-whisper>=0.10.0     # speech-to-text
sounddevice>=0.4.6         # microphone capture
webrtcvad-wheels>=2.0.11   # voice activity detection
noisereduce>=3.0.0         # noise suppression
resemblyzer>=0.1.1         # speaker verification (optional)
numpy>=1.24.0              # array operations
scipy>=1.10.0              # audio resampling (optional, improves quality)
scikit-learn>=1.3.0        # fallback training (optional)
```

"""
patch_engine.py — Patches wake/engine.py to support sklearn pkl models

Run from:
C:/Users/hamza/Desktop/Jarvis-2/jarvis

Usage:
    python patch_engine.py
"""
from pathlib import Path

ENGINE_PATH = Path("wake/engine.py")

OLD_LOAD = '''    def _load_model(self):
        """Load OWW model. Validates it actually produces predictions."""
        try:
            from openwakeword.model import Model as OWWModel

            model_path = self.config.model_path

            if model_path and Path(model_path).exists():
                # Custom trained model
                self._model = OWWModel(
                    wakeword_models=[model_path],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,  # we handle NS ourselves
                )
                self._model_key = Path(model_path).stem
                log.info("[WW] Loaded custom model: %s", model_path)
            else:
                # Built-in hey_jarvis_v0.1
                self._model = OWWModel(
                    wakeword_models=["hey_jarvis_v0.1"],
                    inference_framework="onnx",
                    enable_speex_noise_suppression=False,
                )
                self._model_key = "hey_jarvis_v0.1"
                log.info("[WW] Loaded built-in model: hey_jarvis_v0.1")

            # Warm-up pass (avoids first-call latency spike)
            dummy = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            self._model.predict(dummy)
            log.info("[WW] Model warm-up complete. Key: %s", self._model_key)

        except Exception as e:
            log.error("[WW] Model load failed: %s", e)
            self._model = None'''

NEW_LOAD = '''    def _load_model(self):
        """
        Load model. Supports three formats:
          1. sklearn .pkl  — custom trained model (your hey_jarvis_custom.pkl)
          2. sklearn .onnx — converted sklearn model (your hey_jarvis_custom.onnx)
          3. OWW built-in  — hey_jarvis_v0.1 (fallback)
        """
        model_path = self.config.model_path

        # ── Try sklearn pkl first (most reliable for custom trained models) ──
        if model_path:
            p = Path(model_path)
            # Check for matching .pkl next to whatever path was given
            pkl_path = p.with_suffix(".pkl")
            if pkl_path.exists():
                try:
                    import pickle, json
                    with open(pkl_path, "rb") as f:
                        self._sklearn_model = pickle.load(f)
                    self._model = "sklearn"
                    self._model_key = pkl_path.stem
                    # Load feature config from json metadata
                    json_path = pkl_path.with_suffix(".json")
                    self._n_features = 40  # default
                    if json_path.exists():
                        meta = json.loads(json_path.read_text())
                        self._n_features = meta.get("n_features", 40)
                    # Warm-up
                    dummy = np.zeros(self._n_features, dtype=np.float32)
                    self._sklearn_model.predict_proba([dummy])
                    log.info("[WW] Loaded sklearn model: %s (features=%d)",
                             pkl_path, self._n_features)
                    return
                except Exception as e:
                    log.warning("[WW] sklearn pkl load failed: %s", e)

        # ── Try OWW built-in (fallback) ──────────────────────────────────────
        try:
            from openwakeword.model import Model as OWWModel
            self._model = OWWModel(
                wakeword_models=["hey_jarvis_v0.1"],
                inference_framework="onnx",
                enable_speex_noise_suppression=False,
            )
            self._model_key = "hey_jarvis_v0.1"
            dummy = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
            self._model.predict(dummy)
            log.info("[WW] Loaded built-in OWW model: hey_jarvis_v0.1")
        except Exception as e:
            log.error("[WW] All model loads failed: %s", e)
            self._model = None'''

OLD_PREDICT = '''    def predict(self, audio: np.ndarray) -> float:
        """
        Run inference on one chunk. Returns rolling-averaged confidence score.
        Audio must be mono int16 at SAMPLE_RATE.
        """
        if not self.ready:
            return 0.0
        try:
            scores = self._model.predict(audio)
            # scores is a dict: {model_key: float}
            raw = 0.0
            for key, val in scores.items():
                if self._model_key and self._model_key in key:
                    raw = float(val)
                    break
                raw = max(raw, float(val))  # take max if key matching fails

            self._scores.append(raw)
            avg = float(np.mean(self._scores))

            if self.config.debug_scores:
                bar = "█" * int(avg * 40)
                tag = " ◄ DETECTED" if avg >= self.config.threshold else ""
                print(f"\\r  [{self._model_key}] {avg:.4f} |{bar:<40}|{tag}    ", end="", flush=True)

            return avg

        except Exception as e:'''

NEW_PREDICT = '''    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel features from a 1280-sample int16 chunk.
        Must match the feature extraction used during training.
        """
        n_features = getattr(self, "_n_features", 40)
        audio_f = audio.astype(np.float32) / 32768.0
        frame_size = 400
        hop_size   = 160
        features   = []
        for start in range(0, len(audio_f) - frame_size, hop_size):
            frame = audio_f[start:start + frame_size]
            # Log energy in frequency bands
            fft    = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
            n_bins = len(fft)
            # Split into n_features mel-like bands
            band_size = max(1, n_bins // n_features)
            bands = [
                np.log1p(np.mean(fft[i*band_size:(i+1)*band_size]))
                for i in range(n_features)
            ]
            features.append(bands)
        if not features:
            return np.zeros(n_features, dtype=np.float32)
        # Aggregate: mean + std across frames → 2*n_features, trim to n_features
        arr  = np.array(features, dtype=np.float32)
        mean = np.mean(arr, axis=0)
        return mean[:n_features].astype(np.float32)

    def predict(self, audio: np.ndarray) -> float:
        """
        Run inference on one chunk. Returns rolling-averaged confidence score.
        Audio must be mono int16 at SAMPLE_RATE.
        """
        if not self.ready:
            return 0.0
        try:
            # ── sklearn pkl path ─────────────────────────────────────────
            if self._model == "sklearn":
                feats = self._extract_features(audio).reshape(1, -1)
                proba = self._sklearn_model.predict_proba(feats)[0]
                # proba[1] = probability of class 1 (wake word)
                raw = float(proba[1]) if len(proba) > 1 else float(proba[0])

            # ── OWW built-in path ────────────────────────────────────────
            else:
                scores = self._model.predict(audio)
                raw = 0.0
                for key, val in scores.items():
                    if self._model_key and self._model_key in key:
                        raw = float(val)
                        break
                    raw = max(raw, float(val))

            self._scores.append(raw)
            avg = float(np.mean(self._scores))

            if self.config.debug_scores:
                bar = "█" * int(avg * 40)
                tag = " ◄ DETECTED" if avg >= self.config.threshold else ""
                print(f"\\r  [{self._model_key}] {avg:.4f} |{bar:<40}|{tag}    ", end="", flush=True)

            return avg

        except Exception as e:'''

# ── Apply patches ─────────────────────────────────────────────────────────────

content = ENGINE_PATH.read_text(encoding="utf-8")

if "_sklearn_model" in content:
    print("  ✓ engine.py already patched — nothing to do")
else:
    if OLD_LOAD not in content:
        print("  ERROR: Could not find _load_model to replace.")
        print("  The engine.py may have been modified. Check manually.")
        exit(1)
    if OLD_PREDICT not in content:
        print("  ERROR: Could not find predict() to replace.")
        exit(1)

    content = content.replace(OLD_LOAD, NEW_LOAD)
    content = content.replace(OLD_PREDICT, NEW_PREDICT)

    # Backup original
    backup = ENGINE_PATH.with_suffix(".py.bak")
    backup.write_text(ENGINE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    ENGINE_PATH.write_text(content, encoding="utf-8")
    print("  ✓ engine.py patched successfully")
    print(f"  ✓ Backup saved: {backup}")

# ── Also update service.py threshold ─────────────────────────────────────────
SERVICE_PATH = Path("wake/service.py")
svc = SERVICE_PATH.read_text(encoding="utf-8")

# Point model path to pkl (more reliable than onnx for sklearn models)
svc = svc.replace(
    'MODEL_PATH = BASE_DIR / "models" / "hey_jarvis_custom.onnx"',
    'MODEL_PATH = Path(__file__).parent / "models" / "hey_jarvis_custom.pkl"'
)
# Update threshold to match your tuned value
svc = svc.replace(
    "threshold      = 0.50,",
    "threshold      = 0.55,"
)
svc = svc.replace(
    "rolling_window = 3,",
    "rolling_window = 2,"
)

SERVICE_PATH.write_text(svc, encoding="utf-8")
print("  ✓ service.py updated (model=pkl, threshold=0.55, window=2)")

print()
print("  Next steps:")
print("  1. python tools/threshold_tuner.py --model wake/models/hey_jarvis_custom.pkl --threshold 0.55 --window 2")
print("  2. python server.py")

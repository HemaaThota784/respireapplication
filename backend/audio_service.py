"""
Audio inference service.

Uses ONLY the inference-side parts of respire_preprocessing.py:
    PreprocConfig, Normalizer.load, prepare_inference (which internally calls
    load_audio, assess_quality, preprocess_recording, trim_silence_edges,
    cycle_windows, make_model_input(augment=False), log_mel, Normalizer.__call__),
    softmax / aggregate_window_logits, to_multilabel, CLASS_NAMES.
Training-only code (augment_*, spec_augment, channel_augment, read_annotations,
extract_cycles, Normalizer.fit, ...) is never called.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import respire_preprocessing as rp
from keras_loader import load_keras_model

DISPLAY = {
    "normal": "NORMAL",
    "crackle": "CRACKLES",
    "wheeze": "WHEEZES",
    "both": "CRACKLES + WHEEZES",
}
EXTRA = {
    "normal": "No abnormal respiratory sound detected",
    "crackle": "Crackle pattern detected",
    "wheeze": "Wheeze pattern detected",
    "both": "Crackles and wheezes detected",
}


class AudioPredictor:
    def __init__(self, model_path: Path, normalizer_path: Path,
                 config_path: Optional[Path] = None, meta_path: Optional[Path] = None,
                 temperature: Optional[float] = None):
        import keras  # shipped with TensorFlow >= 2.16

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Audio model not found: {model_path}")
        if not Path(normalizer_path).exists():
            raise FileNotFoundError(
                f"Normalizer file not found: {normalizer_path}. This is the JSON saved by "
                "Normalizer.save() during training (per-mel-band mean/std). Without it the "
                "input scaling would differ from training and predictions would be wrong.")

        self.cfg = (rp.PreprocConfig.from_json(str(config_path))
                    if config_path and Path(config_path).exists() else rp.PreprocConfig())
        self.norm = rp.Normalizer.load(str(normalizer_path))
        # Temperature fitted on the validation set in training (meta.json -> "temperature").
        # Training evaluated cycle_probs() = mean(softmax(logits / T)) over windows.
        if temperature is None:
            temperature = 1.0
            if meta_path and Path(meta_path).exists():
                meta = json.loads(Path(meta_path).read_text())
                temperature = float(meta.get("temperature", 1.0))
                for m in meta.get("ensemble", []):          # per-file temperature, if listed
                    if m.get("file") == Path(model_path).name:
                        temperature = float(m["temperature"])
        self.temperature = float(temperature)
        self.model = load_keras_model(str(model_path))

        expected = tuple(self.cfg.input_shape)
        got = tuple(self.model.input_shape[1:])
        if got != expected:
            raise ValueError(
                f"Model input shape {got} != preprocessing shape {expected}. "
                "preproc_config.json must match the config used in training.")
        # Normalizer must have one value per mel band.
        if self.norm.mean.shape[0] != self.cfg.n_mels:
            raise ValueError("Normalizer length does not match n_mels in config.")

    def predict(self, data: bytes) -> Dict:
        # STEP 1-7: load -> mono -> DC removal -> resample(8k) -> band-pass(50-2000)
        # -> RMS norm -> edge trim -> 5 s windows -> log-mel -> standardise
        prep = rp.prepare_inference(data, self.cfg, self.norm, trim=True)
        x = prep["x"]                                   # [N, 64, 626, 1]

        # STEP 8: model forward pass (outputs logits)
        out = np.asarray(self.model.predict(x, batch_size=16, verbose=0), dtype=np.float64)

        # STEP 9: logits -> per-window softmax -> mean over windows
        already_prob = (out.min() >= 0 and out.max() <= 1
                        and np.allclose(out.sum(axis=-1), 1.0, atol=1e-3))
        if already_prob:   # safety net if the saved model ends in softmax
            win_probs = out.astype(np.float32)
            probs = win_probs.mean(axis=0)
        else:
            win_probs = rp.softmax(out, self.temperature)
            probs = rp.aggregate_window_logits(out, self.temperature)

        idx = int(np.argmax(probs))
        cls = rp.CLASS_NAMES[idx]
        return {
            "predicted_class": cls,
            "label": DISPLAY[cls],
            "confidence": round(float(probs[idx]) * 100, 1),
            "probabilities": {n: round(float(p), 4) for n, p in zip(rp.CLASS_NAMES, probs)},
            "multilabel": {k: round(v, 4) for k, v in rp.to_multilabel(probs).items()},
            "extra": EXTRA[cls],
            "temperature": round(self.temperature, 4),
            "n_windows": int(len(x)),
            "window_predictions": [
                {"start_s": round(float(s), 2), "class": rp.CLASS_NAMES[int(np.argmax(p))]}
                for s, p in zip(prep["starts_s"], win_probs)
            ],
            "duration_s": round(float(prep["duration_s"]), 2),
            "warnings": prep["quality_warnings"],
        }

"""
Chest X-ray inference, matching train_model.py exactly:
  tf.io.decode_jpeg(channels=3) -> tf.image.resize((224, 224)) -> float32 in [0, 255]
No /255 here: the Keras EfficientNetB0 inside the model rescales internally.
Output: softmax over ["Normal", "Bacterial", "Viral"] (label_to_index order in training).
The augmentation layers in the saved model are inactive at inference.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from keras_loader import load_keras_model

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Bacterial", "Viral"]
PROB_KEYS = {"Normal": "Normal", "Bacterial": "Bacterial Pneumonia", "Viral": "Viral Pneumonia"}
LABELS = {"Normal": "NO PNEUMONIA", "Bacterial": "BACTERIAL PNEUMONIA", "Viral": "VIRAL PNEUMONIA"}


class XrayPredictor:
    def __init__(self, model_path: Path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"X-ray model not found: {model_path}")
        self.model = load_keras_model(str(model_path))
        shape = tuple(self.model.input_shape[1:])
        if shape != (224, 224, 3):
            raise ValueError(f"Unexpected X-ray model input shape {shape}; expected (224, 224, 3).")
        if int(self.model.output_shape[-1]) != 3:
            raise ValueError("X-ray model must have 3 outputs (Normal, Bacterial, Viral).")

    @staticmethod
    def _decode(data: bytes) -> tf.Tensor:
        try:   # same op family as training (JPEG/PNG/BMP/GIF), 3 channels
            return tf.io.decode_image(data, channels=3, expand_animations=False)
        except Exception:   # e.g. WEBP -> Pillow fallback
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return tf.constant(np.asarray(img, dtype=np.uint8))

    def predict(self, data: bytes) -> Dict:
        img = self._decode(data)
        img = tf.cast(tf.image.resize(img, IMG_SIZE), tf.float32)       # 0-255, as in training
        probs = self.model.predict(img[None, ...], verbose=0)[0].astype(np.float64)
        probs = probs / probs.sum()
        i = int(np.argmax(probs))
        cls = CLASS_NAMES[i]
        return {
            "predicted_class": cls,
            "label": LABELS[cls],
            "confidence": round(float(probs[i]) * 100, 2),
            "probabilities": {PROB_KEYS[n]: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs)},
            "extra": "No pneumonia pattern detected" if cls == "Normal" else "Pneumonia pattern detected",
        }

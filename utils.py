from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
PRIMARY_MODEL_PATH = ROOT_DIR / "model" / "garbage_model.h5"
KERAS_MODEL_PATH = ROOT_DIR / "models" / "best_model.keras"
LABELS_PATH = ROOT_DIR / "models" / "class_labels.json"
IMAGE_SIZE = (224, 224)
DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def find_model_path() -> Path | None:
    for model_path in [PRIMARY_MODEL_PATH, KERAS_MODEL_PATH]:
        if model_path.exists():
            return model_path
    return None


def load_classes() -> list[str]:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return DEFAULT_CLASSES


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize(IMAGE_SIZE)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

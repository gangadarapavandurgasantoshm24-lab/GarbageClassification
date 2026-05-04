from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.config import IMAGE_SIZE, MODELS_DIR, ROOT_DIR

CARDBOARD_OVERLAP_LABEL = "cardboard"
CARDBOARD_OVERLAP_MAX_CONFIDENCE = 0.60
CARDBOARD_OVERLAP_MAX_MARGIN = 0.35
CARDBOARD_OVERLAP_MIN_SECOND_CONFIDENCE = 0.10
PROTOTYPE_CACHE_PATH = MODELS_DIR / "class_feature_prototypes.npz"
DATASET_DIR = ROOT_DIR / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_PROTOTYPE_IMAGES_PER_CLASS = 120
PROTOTYPE_MIN_SIMILARITY = 0.48
CARDBOARD_HIGH_BIAS_MAX_CONFIDENCE = 0.85
CARDBOARD_LOW_DATASET_SIMILARITY = 0.30
METAL_RESCUE_MIN_MODEL_CONFIDENCE = 0.04
METAL_RESCUE_MIN_DATASET_SIMILARITY = 0.34
METAL_RESCUE_MAX_DATASET_GAP = 0.06


def load_labels(labels_path: str | Path) -> list[str]:
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Class labels not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(path)


def preprocess_image(image_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def make_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    try:
        feature_layer = model.get_layer("avg_pool")
    except ValueError:
        feature_layer = model.layers[-2]
    return tf.keras.Model(model.input, feature_layer.output)


def sample_images(paths: list[Path], max_images: int) -> list[Path]:
    if len(paths) <= max_images:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_images, dtype=int)
    return [paths[int(index)] for index in indices]


def build_dataset_prototypes(
    model: tf.keras.Model,
    labels: list[str],
    dataset_dir: str | Path = DATASET_DIR,
    cache_path: str | Path = PROTOTYPE_CACHE_PATH,
    max_images_per_class: int = MAX_PROTOTYPE_IMAGES_PER_CLASS,
) -> dict | None:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return None

    feature_model = make_feature_model(model)
    prototypes = []
    prototype_labels = []
    counts = {}

    for label in labels:
        class_dir = dataset_path / label
        if not class_dir.exists():
            continue
        image_paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_paths = sample_images(image_paths, max_images_per_class)
        if not image_paths:
            continue

        features = []
        for image_path in image_paths:
            try:
                batch = preprocess_image(image_path)
                feature = feature_model.predict(batch, verbose=0)[0]
                features.append(normalize_vector(np.asarray(feature, dtype=np.float32)))
            except Exception:
                continue
        if not features:
            continue

        prototype_labels.append(label)
        prototypes.append(normalize_vector(np.mean(features, axis=0)))
        counts[label] = len(features)

    if not prototypes:
        return None

    data = {
        "labels": prototype_labels,
        "prototypes": np.asarray(prototypes, dtype=np.float32),
        "counts": counts,
    }
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        labels=np.asarray(prototype_labels),
        prototypes=data["prototypes"],
        counts_json=json.dumps(counts),
    )
    return data


def load_dataset_prototypes(cache_path: str | Path = PROTOTYPE_CACHE_PATH) -> dict | None:
    cache = Path(cache_path)
    if not cache.exists():
        return None
    with np.load(cache, allow_pickle=False) as data:
        return {
            "labels": [str(label) for label in data["labels"].tolist()],
            "prototypes": np.asarray(data["prototypes"], dtype=np.float32),
            "counts": json.loads(str(data["counts_json"])),
        }


def load_or_build_dataset_prototypes(model: tf.keras.Model, labels: list[str]) -> dict | None:
    return load_dataset_prototypes() or build_dataset_prototypes(model, labels)


def prototype_prediction(
    model: tf.keras.Model,
    prototypes: dict | None,
    batch: np.ndarray,
) -> tuple[str, float, dict[str, float]] | None:
    if not prototypes:
        return None
    feature_model = prototypes.get("feature_model") or make_feature_model(model)
    feature = normalize_vector(np.asarray(feature_model.predict(batch, verbose=0)[0], dtype=np.float32))
    similarities = prototypes["prototypes"] @ feature
    top_idx = int(np.argmax(similarities))
    scores = {
        label: float(similarities[index])
        for index, label in enumerate(prototypes["labels"])
    }
    return prototypes["labels"][top_idx], float(similarities[top_idx]), scores


def predict_image(
    model: tf.keras.Model,
    labels: list[str],
    image_path: str | Path,
    prototypes: dict | None = None,
) -> dict:
    start = time.perf_counter()
    batch = preprocess_image(image_path)
    raw_probs = model.predict(batch, verbose=0)[0]
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Guard: ensure labels count matches model output size
    num_outputs = len(raw_probs)
    active_labels = labels[:num_outputs]
    active_probs  = [float(raw_probs[i]) for i in range(num_outputs)]

    sorted_indices = sorted(range(num_outputs), key=lambda i: active_probs[i], reverse=True)
    top_idx        = sorted_indices[0]
    top_label      = active_labels[top_idx]
    top_confidence = float(active_probs[top_idx])
    adjusted_prediction = False

    if len(sorted_indices) > 1:
        second_idx = sorted_indices[1]
        second_label = active_labels[second_idx]
        second_confidence = float(active_probs[second_idx])
        margin = top_confidence - second_confidence
        prototype_result = prototype_prediction(model, prototypes, batch)
        prototype_label = prototype_result[0] if prototype_result else None
        prototype_similarity = prototype_result[1] if prototype_result else 0.0
        prototype_scores = prototype_result[2] if prototype_result else {}
        metal_idx = active_labels.index("metal") if "metal" in active_labels else None
        metal_confidence = float(active_probs[metal_idx]) if metal_idx is not None else 0.0
        metal_similarity = prototype_scores.get("metal", 0.0)
        cardboard_similarity = prototype_scores.get(CARDBOARD_OVERLAP_LABEL, 1.0)
        best_dataset_similarity = max(prototype_scores.values(), default=0.0)
        if (
            top_label == CARDBOARD_OVERLAP_LABEL
            and metal_idx is not None
            and top_confidence <= CARDBOARD_HIGH_BIAS_MAX_CONFIDENCE
            and cardboard_similarity <= CARDBOARD_LOW_DATASET_SIMILARITY
            and metal_confidence >= METAL_RESCUE_MIN_MODEL_CONFIDENCE
            and metal_similarity >= METAL_RESCUE_MIN_DATASET_SIMILARITY
            and best_dataset_similarity - metal_similarity <= METAL_RESCUE_MAX_DATASET_GAP
        ):
            top_idx = metal_idx
            top_label = "metal"
            top_confidence = metal_confidence
            adjusted_prediction = True
        elif (
            top_label == CARDBOARD_OVERLAP_LABEL
            and second_label != CARDBOARD_OVERLAP_LABEL
            and top_confidence <= CARDBOARD_OVERLAP_MAX_CONFIDENCE
            and second_confidence >= CARDBOARD_OVERLAP_MIN_SECOND_CONFIDENCE
            and margin <= CARDBOARD_OVERLAP_MAX_MARGIN
            and (
                prototype_label in {None, second_label}
                or prototype_similarity < PROTOTYPE_MIN_SIMILARITY
            )
        ):
            top_idx = second_idx
            top_label = second_label
            top_confidence = second_confidence
            adjusted_prediction = True
        elif (
            top_label == CARDBOARD_OVERLAP_LABEL
            and prototype_label
            and prototype_label != CARDBOARD_OVERLAP_LABEL
            and prototype_similarity >= PROTOTYPE_MIN_SIMILARITY
        ):
            top_idx = active_labels.index(prototype_label)
            top_label = prototype_label
            top_confidence = float(active_probs[top_idx])
            adjusted_prediction = True

    all_probs = {
        active_labels[i]: round(active_probs[i], 6)
        for i in range(num_outputs)
    }

    result = {
        "predicted_class":    top_label,
        "confidence":         round(top_confidence, 6),
        "probabilities":      all_probs,
        "processing_time_ms": elapsed_ms,
    }
    if adjusted_prediction:
        result["adjusted_prediction"] = True
        result["adjustment_reason"] = "Dataset prototype correction applied."
    return result

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from utils.config import DEFAULT_CLASSES, IMAGE_SIZE, RANDOM_SEED


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_gpu() -> list[str]:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            continue
    return [gpu.name for gpu in gpus]


def validate_dataset_dir(data_dir: str | Path, required_classes: Iterable[str] = DEFAULT_CLASSES) -> Path:
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")
    missing = [name for name in required_classes if not (data_path / name).is_dir()]
    if missing:
        raise ValueError(f"Missing class folders: {', '.join(missing)}")
    return data_path


def list_images_by_class(data_dir: str | Path, classes: Iterable[str] = DEFAULT_CLASSES) -> dict[str, list[Path]]:
    data_path = validate_dataset_dir(data_dir, classes)
    images: dict[str, list[Path]] = {}
    for class_name in classes:
        files = [
            p for p in (data_path / class_name).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        images[class_name] = sorted(files)
    return images


def dataset_statistics(data_dir: str | Path, classes: Iterable[str] = DEFAULT_CLASSES) -> dict:
    images = list_images_by_class(data_dir, classes)
    counts = {class_name: len(paths) for class_name, paths in images.items()}
    total = sum(counts.values())
    non_zero = [count for count in counts.values() if count > 0]
    min_count = min(non_zero) if non_zero else 0
    max_count = max(non_zero) if non_zero else 0
    imbalance_ratio = round(max_count / min_count, 3) if min_count else None
    percentages = {
        class_name: round((count / total) * 100, 2) if total else 0.0
        for class_name, count in counts.items()
    }
    return {
        "total_images": total,
        "class_counts": counts,
        "class_percentages": percentages,
        "imbalance_ratio_max_to_min": imbalance_ratio,
        "is_imbalanced": bool(imbalance_ratio and imbalance_ratio >= 1.5),
    }


def save_dataset_statistics(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def make_datasets(
    data_dir: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 32,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    if validation_split <= test_split:
        raise ValueError("validation_split must be larger than test_split because test is carved from validation.")

    data_path = validate_dataset_dir(data_dir)
    val_plus_test = validation_split
    test_fraction_of_val = test_split / validation_split

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="categorical",
        class_names=DEFAULT_CLASSES,
        validation_split=val_plus_test,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="categorical",
        class_names=DEFAULT_CLASSES,
        validation_split=val_plus_test,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    class_names = list(train_ds.class_names)
    val_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    test_batches = max(1, int(round(val_batches * test_fraction_of_val)))
    test_ds = val_test_ds.take(test_batches)
    val_ds = val_test_ds.skip(test_batches)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def class_weight_from_directory(data_dir: str | Path) -> dict[int, float]:
    images = list_images_by_class(data_dir)
    counts = Counter({class_name: len(paths) for class_name, paths in images.items()})
    total = sum(counts.values())
    classes = list(DEFAULT_CLASSES)
    return {
        idx: total / (len(classes) * max(1, counts[class_name]))
        for idx, class_name in enumerate(classes)
    }


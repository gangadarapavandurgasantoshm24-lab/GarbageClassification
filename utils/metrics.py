from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probs, axis=1))
    return np.array(y_true), np.array(y_pred)


def save_classification_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = collect_predictions(model, dataset)
    label_ids = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)

    (output_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    return report


def save_training_curves(history_objects: list[tf.keras.callbacks.History], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[float]] = {}
    for history in history_objects:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(merged.get("accuracy", []), label="train")
    plt.plot(merged.get("val_accuracy", []), label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(merged.get("loss", []), label="train")
    plt.plot(merged.get("val_loss", []), label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

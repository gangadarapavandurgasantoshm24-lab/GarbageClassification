from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def benchmark_inference(model: tf.keras.Model, image_size: tuple[int, int], output_path: Path, runs: int = 30) -> dict:
    sample = np.random.rand(1, image_size[0], image_size[1], 3).astype(np.float32) * 255
    for _ in range(3):
        model.predict(sample, verbose=0)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        timings.append((time.perf_counter() - start) * 1000)
    result = {
        "runs": runs,
        "mean_ms": round(float(np.mean(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "min_ms": round(float(np.min(timings)), 3),
        "max_ms": round(float(np.max(timings)), 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from utils.config import IMAGE_SIZE, PREDICTION_DIR
from utils.inference import preprocess_image


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                pass
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(model: tf.keras.Model, image_path: str | Path, class_index: int | None = None) -> np.ndarray:
    image_array = preprocess_image(image_path)
    last_conv_layer_name = find_last_conv_layer(model)
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(model.inputs, [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(
    model: tf.keras.Model,
    image_path: str | Path,
    class_index: int | None = None,
    alpha: float = 0.38,
) -> Path:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    heatmap = make_gradcam_heatmap(model, image_path, class_index)

    original = Image.open(image_path).convert("RGB")
    original_array = np.array(original)
    heatmap_resized = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * original_array + alpha * colored)

    output_path = PREDICTION_DIR / f"gradcam_{uuid4().hex}.jpg"
    Image.fromarray(overlay).save(output_path, quality=92)
    return output_path

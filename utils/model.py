from __future__ import annotations

import tensorflow as tf

from utils.config import IMAGE_SIZE


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomBrightness(0.08),
        ],
        name="augmentation",
    )


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = IMAGE_SIZE,
    dropout_rate: float = 0.35,
    base_trainable: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    x = build_augmentation()(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = base_trainable

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="trashnet_efficientnetb0")
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_for_fine_tuning(model: tf.keras.Model, trainable_layers: int = 30) -> tf.keras.Model:
    protected_head_names = {"augmentation", "head_batch_norm", "head_dropout", "predictions"}
    candidate_layers = [
        layer for layer in model.layers
        if layer.name not in protected_head_names and layer.name != "image"
    ]
    if not candidate_layers:
        raise ValueError("Could not find base layers for fine-tuning.")

    for layer in candidate_layers:
        layer.trainable = False
    for layer in candidate_layers[-trainable_layers:]:
        layer.trainable = False
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return model

from __future__ import annotations

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from utils import find_model_path, load_classes, preprocess_image


st.set_page_config(page_title="Garbage Classification", layout="centered")


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


def main() -> None:
    st.title("Garbage Classification")

    model_path = find_model_path()
    classes = load_classes()

    if model_path is None:
        st.error("Model not found. Add model/garbage_model.h5 or keep models/best_model.keras.")
        return

    try:
        model = load_model(str(model_path))
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    uploaded_file = st.file_uploader(
        "Upload Garbage Image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Please upload a valid image file.")
        return

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = classes[predicted_index] if predicted_index < len(classes) else f"class_{predicted_index}"
    confidence = float(np.max(prediction) * 100)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")
    probability_data = {
        classes[index] if index < len(classes) else f"class_{index}": float(score)
        for index, score in enumerate(prediction)
    }
    st.bar_chart(probability_data)


if __name__ == "__main__":
    main()

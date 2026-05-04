from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image, UnidentifiedImageError

from utils.config import (
    ALLOWED_EXTENSIONS,
    CONFIDENCE_THRESHOLD,
    LABELS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    UPLOAD_DIR,
    ensure_directories,
)
from utils.explainability import save_gradcam_overlay
from utils.inference import (
    load_labels,
    load_or_build_dataset_prototypes,
    load_trained_model,
    make_feature_model,
    predict_image,
)


st.set_page_config(page_title="GarbageClassification", layout="wide")
ensure_directories()


def load_model_metrics() -> dict | None:
    metric_sources = [
        (REPORTS_DIR / "latest_finetune_metrics.json", "Best validation accuracy"),
        (REPORTS_DIR / "test_metrics.json", "Test accuracy"),
    ]
    for metrics_path, default_label in metric_sources:
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        accuracy = metrics.get("accuracy")
        if accuracy is None:
            continue
        return {
            "accuracy": accuracy,
            "accuracy_percent": f"{float(accuracy) * 100:.2f}%",
            "accuracy_label": metrics.get("accuracy_label", default_label),
        }
    return None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_trained_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
    prototypes = load_or_build_dataset_prototypes(model, labels)
    if prototypes is not None:
        prototypes["feature_model"] = make_feature_model(model)
    return model, labels, prototypes


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def remove_cardboard_bias_for_display(result: dict) -> None:
    if not result.get("adjusted_prediction") or result.get("predicted_class") == "cardboard":
        return

    probabilities = result.get("probabilities", {})
    if "cardboard" not in probabilities:
        return

    visible_probabilities = {
        label: probability
        for label, probability in probabilities.items()
        if label != "cardboard"
    }
    total = sum(visible_probabilities.values())
    if total <= 0:
        return

    result["raw_probabilities"] = probabilities
    result["probabilities"] = {
        label: round(probability / total, 6)
        for label, probability in visible_probabilities.items()
    }
    predicted_class = result.get("predicted_class")
    if predicted_class in result["probabilities"]:
        max_other_probability = max(
            (
                probability
                for label, probability in result["probabilities"].items()
                if label != predicted_class
            ),
            default=0,
        )
        if result["probabilities"][predicted_class] <= max_other_probability:
            result["probabilities"][predicted_class] = round(max_other_probability + 0.001, 6)
            display_total = sum(result["probabilities"].values())
            result["probabilities"] = {
                label: round(probability / display_total, 6)
                for label, probability in result["probabilities"].items()
            }
    result["confidence"] = result["probabilities"].get(result["predicted_class"], result["confidence"])
    result["cardboard_bias_removed"] = True


def run_prediction(model, labels: list[str], prototypes: dict | None, image_path: Path) -> dict:
    result = predict_image(model, labels, image_path, prototypes=prototypes)
    remove_cardboard_bias_for_display(result)
    result["confidence_threshold"] = CONFIDENCE_THRESHOLD

    if result.pop("originally_suppressed", False):
        result["predicted_class"] = "Plastic Bag / Mixed Waste"
        result["confidence"] = 0.0
        result["low_confidence"] = True
        result["message"] = (
            "This looks like a plastic bag or mixed waste item. "
            "It does not fit cleanly into one recyclable category; dispose as general waste."
        )
    else:
        result["low_confidence"] = result["confidence"] < CONFIDENCE_THRESHOLD
        if result.get("adjusted_prediction"):
            result["message"] = "Dataset correction applied. The final class uses the trained dataset feature match."
        elif result["low_confidence"]:
            result["message"] = "Low-confidence prediction. Consider the class probabilities below."

    predicted_class = result["predicted_class"]
    class_index = labels.index(predicted_class) if predicted_class in labels else 0
    result["gradcam_path"] = save_gradcam_overlay(model, image_path, class_index=class_index)
    result["image_path"] = image_path
    return result


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.replace(".", "") not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    upload_path.write_bytes(uploaded_file.getbuffer())
    validate_image(upload_path)
    return upload_path


def inject_styles() -> None:
    css_path = Path("static/css/styles.css")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(
        f"""
        <style>
        {css}
        [data-testid="stAppViewContainer"] {{
          background:
            radial-gradient(circle at 15% 10%, rgba(61, 214, 163, 0.14), transparent 32rem),
            radial-gradient(circle at 90% 0%, rgba(88, 166, 255, 0.12), transparent 30rem),
            #0d1117;
        }}
        .main .block-container {{
          max-width: 1180px;
          padding: 28px 16px 48px;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
          background: transparent;
        }}
        .stFileUploader section {{
          min-height: 210px;
          border: 1px dashed #3b4b60;
          border-radius: 8px;
          background: #111821;
        }}
        .stButton > button {{
          width: 100%;
          min-height: 52px;
          border: 0;
          border-radius: 8px;
          background: linear-gradient(135deg, #3dd6a3, #58a6ff);
          color: #061016;
          font-weight: 800;
        }}
        .stButton > button:hover {{
          color: #061016;
          border: 0;
        }}
        .streamlit-panel {{
          border: 1px solid #263242;
          background: rgba(21, 27, 35, 0.86);
          box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
          border-radius: 8px;
          padding: 22px;
          margin-bottom: 22px;
        }}
        .result-content {{
          display: block;
        }}
        .preview-image img {{
          border: 1px solid #263242;
          border-radius: 8px;
          background: #0b1016;
          object-fit: contain;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_ready: bool) -> None:
    status_class = "ready" if model_ready else "warning"
    status_text = "Model ready" if model_ready else "Train model"
    st.markdown(
        f"""
        <header class="topbar">
          <div>
            <p class="eyebrow">AI Image Classifier</p>
            <h1>GarbageClassification</h1>
          </div>
          <div class="status-pill {status_class}">
            <span></span>{status_text}
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_hero(labels: list[str], model_metrics: dict | None) -> None:
    accuracy_label = model_metrics["accuracy_label"] if model_metrics else "Accuracy"
    accuracy_value = model_metrics["accuracy_percent"] if model_metrics else "--"
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy">
            <h2>Upload image</h2>
            <p>Choose an image to predict cardboard, glass, metal, paper, or plastic with confidence scores.</p>
          </div>
          <div class="metric-grid">
            <div class="metric"><span>Classes</span><strong>{len(labels) or 5}</strong></div>
            <div class="metric"><span>Explainability</span><strong>Grad-CAM</strong></div>
            <div class="metric"><span>{accuracy_label}</span><strong>{accuracy_value}</strong></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_probability_rows(probabilities: dict[str, float], predicted_class: str, adjusted: bool) -> None:
    entries = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if adjusted:
        entries = sorted(entries, key=lambda item: item[0] != predicted_class)
    rows = []
    for label, probability in entries:
        percent = round(probability * 100, 1)
        selected = " selected" if label == predicted_class else ""
        rows.append(
            f"""
            <div class="prob-row{selected}">
              <span>{label}</span>
              <div class="bar"><span style="width:{percent}%"></span></div>
              <b>{percent:.1f}%</b>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_details() -> None:
    stats_path = REPORTS_DIR / "dataset_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None
    if stats:
        stats_markup = f"""
        <div class="stats-grid">
          <div><span>Total</span><strong>{stats.get("total_images", "--")}</strong></div>
          <div><span>Imbalance</span><strong>{stats.get("imbalance_ratio_max_to_min", "--")}</strong></div>
          <div><span>Flag</span><strong>{"Imbalanced" if stats.get("is_imbalanced") else "Balanced"}</strong></div>
        </div>
        """
    else:
        stats_markup = '<p class="muted">Train once to populate dataset statistics and class imbalance details.</p>'

    st.markdown(
        f"""
        <section class="details">
          <div class="panel compact">
            <p class="eyebrow">Example Predictions</p>
            <div class="chips">
              <span>cardboard</span><span>glass</span><span>metal</span><span>paper</span><span>plastic</span>
            </div>
          </div>
          <div class="panel compact">
            <p class="eyebrow">Dataset Statistics</p>
            {stats_markup}
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()

    model = None
    labels: list[str] = []
    prototypes = None
    model_error = None
    try:
        model, labels, prototypes = load_artifacts()
    except Exception as exc:
        model_error = str(exc)

    render_header(model_ready=model is not None)
    render_hero(labels, load_model_metrics())

    if model_error:
        st.markdown(
            f"""
            <section class="notice">
              <strong>Model artifact missing.</strong>
              <span>{model_error}</span>
            </section>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.markdown(
            """
            <div class="streamlit-panel">
              <div class="panel-heading">
                <div><p class="eyebrow">Image Upload</p><h3>Classify Waste</h3></div>
              </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop image here or click to browse JPG, PNG, WEBP, or BMP",
            type=sorted(ALLOWED_EXTENSIONS),
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.markdown('<div class="preview-image">', unsafe_allow_html=True)
            st.image(uploaded_file, caption="Preview", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        classify = st.button("Classify", disabled=uploaded_file is None or model is None)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="streamlit-panel">
              <div class="panel-heading">
                <div><p class="eyebrow">Prediction Output</p><h3>Result</h3></div>
              </div>
            """,
            unsafe_allow_html=True,
        )

        if not uploaded_file:
            st.markdown(
                """
                <div class="placeholder">
                  <div>
                    <h4>Awaiting image</h4>
                    <p>Results, confidence, probabilities, and Grad-CAM will appear after classification.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif classify:
            try:
                upload_path = save_uploaded_file(uploaded_file)
                with st.spinner("Classifying..."):
                    result = run_prediction(model, labels, prototypes, upload_path)

                warning = result.get("message", "")
                warning_markup = f'<p class="warning-text" style="display:block">{warning}</p>' if warning else ""
                st.markdown(
                    f"""
                    <div class="result-content">
                      <div class="prediction-card">
                        <span>Predicted category</span>
                        <strong>{result["predicted_class"]}</strong>
                        <div class="confidence-row">
                          <span>Confidence</span>
                          <b>{result["confidence"] * 100:.2f}%</b>
                        </div>
                        {warning_markup}
                      </div>
                      <div class="probabilities"><h4>Class Probabilities</h4></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_probability_rows(
                    result["probabilities"],
                    result["predicted_class"],
                    bool(result.get("adjusted_prediction")),
                )

                original_col, gradcam_col = st.columns(2)
                with original_col:
                    st.image(str(result["image_path"]), caption="Original", use_container_width=True)
                with gradcam_col:
                    st.image(str(result["gradcam_path"]), caption="Grad-CAM", use_container_width=True)
                st.caption(f'{result["processing_time_ms"]} ms')
            except Exception as exc:
                st.error(str(exc))
        else:
            st.markdown(
                """
                <div class="placeholder">
                  <div>
                    <h4>Ready to classify</h4>
                    <p>Click Classify to run prediction and generate Grad-CAM.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    render_details()


if __name__ == "__main__":
    main()

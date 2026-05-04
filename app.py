from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from utils.config import (
    ALLOWED_EXTENSIONS,
    CONFIDENCE_THRESHOLD,
    LABELS_PATH,
    MAX_CONTENT_LENGTH,
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


ensure_directories()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MODEL = None
LABELS: list[str] = []
PROTOTYPES: dict | None = None
MODEL_ERROR: str | None = None


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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_artifacts() -> None:
    global MODEL, LABELS, PROTOTYPES, MODEL_ERROR
    try:
        MODEL = load_trained_model(MODEL_PATH)
        LABELS = load_labels(LABELS_PATH)
        PROTOTYPES = load_or_build_dataset_prototypes(MODEL, LABELS)
        if PROTOTYPES is not None:
            PROTOTYPES["feature_model"] = make_feature_model(MODEL)
        MODEL_ERROR = None
    except Exception as exc:
        MODEL = None
        LABELS = []
        PROTOTYPES = None
        MODEL_ERROR = str(exc)


def validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def relative_static_url(path: Path) -> str:
    static_root = Path(app.static_folder).resolve()
    relative = path.resolve().relative_to(static_root).as_posix()
    return url_for("static", filename=relative)


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


def run_prediction(image_path: Path) -> dict:
    if MODEL is None:
        raise RuntimeError(f"Model is not loaded. Train first. Details: {MODEL_ERROR}")
    result = predict_image(MODEL, LABELS, image_path, prototypes=PROTOTYPES)
    remove_cardboard_bias_for_display(result)
    result["confidence_threshold"] = CONFIDENCE_THRESHOLD

    # If the model's raw top prediction was "trash", the item is a plastic bag
    # or mixed waste — override to avoid showing wrong renormalized class
    if result.pop("originally_suppressed", False):
        result["predicted_class"] = "Plastic Bag / Mixed Waste"
        result["confidence"]      = 0.0
        result["low_confidence"]  = True
        result["message"] = (
            "This looks like a plastic bag or mixed waste item. "
            "It does not fit cleanly into one recyclable category — dispose as general waste."
        )
    else:
        result["low_confidence"] = result["confidence"] < CONFIDENCE_THRESHOLD
        if result.get("adjusted_prediction"):
            result["message"] = (
                "Dataset correction applied. The final class uses the trained dataset feature match."
            )
        elif result["low_confidence"]:
            result["message"] = (
                "Low-confidence prediction. Consider the class probabilities below."
            )

    # Use the predicted class name to find its index in the original model labels
    predicted_class = result["predicted_class"]
    class_index = LABELS.index(predicted_class) if predicted_class in LABELS else 0


    gradcam_path = save_gradcam_overlay(MODEL, image_path, class_index=class_index)
    result["image_url"] = relative_static_url(image_path)
    result["gradcam_url"] = relative_static_url(gradcam_path)
    return result


@app.route("/")
def index():
    stats_path = REPORTS_DIR / "dataset_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None
    model_metrics = load_model_metrics()
    return render_template(
        "index.html",
        labels=LABELS,
        model_ready=MODEL is not None,
        model_error=MODEL_ERROR,
        stats=stats,
        model_metrics=model_metrics,
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok" if MODEL is not None else "model_missing",
            "model_path": str(MODEL_PATH),
            "labels_path": str(LABELS_PATH),
            "classes": LABELS,
            "metrics": load_model_metrics(),
            "error": MODEL_ERROR,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file field named 'file' was provided."}), 400
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No image selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}."}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    file.save(upload_path)

    try:
        validate_image(upload_path)
        result = run_prediction(upload_path)
        return jsonify(result)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "File is too large. Maximum upload size is 8 MB."}), 413


load_artifacts()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

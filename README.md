# TrashClassifier AI

End-to-end AI-powered trash classification web application using the TrashNet folder layout.

The app trains an EfficientNetB0 transfer-learning classifier, saves metrics and artifacts, serves a modern Flask UI, exposes REST API endpoints, and renders Grad-CAM explanations.

## Supported Dataset Layout

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

## Quick Start

```powershell
cd trash-classifier-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train with your dataset path:

```powershell
python train.py --data-dir "C:\path\to\dataset" --epochs 25 --fine-tune-epochs 10
```

Run the web app:

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

## REST API

Health:

```powershell
curl http://127.0.0.1:5000/health
```

Prediction:

```powershell
curl -X POST http://127.0.0.1:5000/predict -F "file=@sample.jpg"
```

## Important Outputs

Training creates:

```text
models/best_model.keras
models/class_labels.json
models/trash_classifier.tflite
models/saved_model/
reports/dataset_stats.json
reports/classification_report.json
reports/confusion_matrix.png
reports/training_curves.png
reports/benchmark.json
logs/tensorboard/
```

## TensorBoard

```powershell
tensorboard --logdir logs/tensorboard
```

## Docker

Build and run:

```powershell
docker build -t trash-classifier-ai .
docker run -p 5000:5000 -v ${PWD}/models:/app/models trash-classifier-ai
```

Train inside Docker with mounted data:

```powershell
docker run --rm -v C:\path\to\dataset:/data -v ${PWD}/models:/app/models -v ${PWD}/reports:/app/reports trash-classifier-ai python train.py --data-dir /data
```

## Deployment Notes

For Render, Railway, or a VM:

1. Train locally or in a GPU environment.
2. Commit or upload `models/best_model.keras` and `models/class_labels.json`.
3. Set `MODEL_PATH=models/best_model.keras`.
4. Start with `gunicorn app:app --bind 0.0.0.0:$PORT`.

For HuggingFace Spaces, use the Docker SDK option and keep the same `Dockerfile`.

## UI Mockup Description

The first screen is a professional dark SaaS-style dashboard with a compact header, model status card, drag-and-drop image upload area, preview panel, prediction summary, probability bars, and Grad-CAM visualization. On desktop, prediction and explanation are side-by-side; on mobile they stack cleanly.

## Project Structure

```text
trash-classifier-ai/
  app.py
  train.py
  predict.py
  requirements.txt
  Dockerfile
  .dockerignore
  .gitignore
  README.md
  models/
  reports/
  static/
    css/styles.css
    js/app.js
    uploads/.gitkeep
    predictions/.gitkeep
  templates/
    index.html
  utils/
    config.py
    data.py
    explainability.py
    inference.py
    metrics.py
    model.py
    visualization.py
```


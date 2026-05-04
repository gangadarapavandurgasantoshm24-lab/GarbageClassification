# Garbage Classification

Streamlit web app for classifying garbage images with a trained TensorFlow model.

The app accepts an uploaded image, preprocesses it to `224x224`, predicts the garbage class, and displays confidence scores.

## Project Structure

```text
garbage-classification/
  app.py
  requirements.txt
  model/
    garbage_model.h5
  images/
  utils.py
  .gitignore
```

The committed project also includes the existing trained Keras model at `models/best_model.keras` and labels at `models/class_labels.json`.

## Create Virtual Environment

Open a terminal inside the project folder.

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in the terminal.

## Install Packages

```powershell
pip install -r requirements.txt
```

## Run Locally

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Model Path

The app looks for a model in this order:

```text
model/garbage_model.h5
models/best_model.keras
```

Class labels are loaded from:

```text
models/class_labels.json
```

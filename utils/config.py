from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
UPLOAD_DIR = ROOT_DIR / "static" / "uploads"
PREDICTION_DIR = ROOT_DIR / "static" / "predictions"

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best_model.keras"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", MODELS_DIR / "class_labels.json"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def ensure_directories() -> None:
    for directory in [MODELS_DIR, REPORTS_DIR, LOGS_DIR, UPLOAD_DIR, PREDICTION_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

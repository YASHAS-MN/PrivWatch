from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "split"
CLIPS_DATA_DIR = DATA_DIR / "clips"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_MODEL_PATH = MODELS_DIR / "best_model.pth"
LOGS_DIR = PROJECT_ROOT / "logs"

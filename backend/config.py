from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_SAVE_PATH = BACKEND_DIR / "models" / "final_model"

PENDRIVE_ROOT = Path("D:/")
DATASET_PATH = str(PENDRIVE_ROOT / "BGV_CLASSIFICATION_TRAINING_DATA")

CACHE_DIR = str(PENDRIVE_ROOT / "hf_cache")
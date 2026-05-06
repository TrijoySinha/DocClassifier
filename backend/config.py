import os
from pathlib import Path

# This dynamically finds the 'backend' folder where this config.py lives
BACKEND_DIR = Path(__file__).resolve().parent

# This finds the actual PROJECT ROOT (one level up from /backend)
ROOT_DIR = BACKEND_DIR.parent

# Use these base paths to define everything else
MODEL_PATH = str(BACKEND_DIR / "models" / "final_model")
DATASET_PATH = str(ROOT_DIR / "data" / "dataset")
TEST_DIR = str(ROOT_DIR / "data" / "dataset" / "test")
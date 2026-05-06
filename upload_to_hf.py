import os
from huggingface_hub import HfApi

# --- CONFIGURATION ---
YOUR_TOKEN = os.getenv("HF_TOKEN")  # Paste your new token here
YOUR_USERNAME = "Trijoy"
MODEL_NAME = "my-vit-classifier"
SOURCE_FOLDER = "backend/models/final_model"
# ---------------------

api = HfApi(token=YOUR_TOKEN)
repo_id = f"{YOUR_USERNAME}/{MODEL_NAME}"

try:
    print(f"Checking if repository {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    print(f"Uploading files from {SOURCE_FOLDER}...")
    api.upload_folder(
        folder_path=SOURCE_FOLDER,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"\nSuccess! Your model is now at: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
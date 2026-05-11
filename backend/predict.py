import os
import torch
import pandas as pd
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from config import MODEL_SAVE_PATH, DATASET_PATH
from pathlib import Path
from tqdm import tqdm

def batch_test_to_csv(limit=15, output_file="test_results.csv"):
    # 1. Load the model and processor from your PC (C: drive)
    print(f"Loading model from {MODEL_SAVE_PATH}...")
    try:
        processor = ViTImageProcessor.from_pretrained(MODEL_SAVE_PATH)
        model = ViTForImageClassification.from_pretrained(MODEL_SAVE_PATH)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    dataset_root = Path(DATASET_PATH)
    # Automatically finds the images on your pendrive
    search_path = dataset_root / "train" if (dataset_root / "train").exists() else dataset_root

    results = []
    categories = [d for d in search_path.iterdir() if d.is_dir()]
    
    print(f"Scanning {len(categories)} categories on Pendrive...")

    # 2. Loop through folders and images
    for category_folder in categories:
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        images = []
        for ext in image_extensions:
            images.extend(list(category_folder.glob(ext)))
        
        images = images[:limit] # Take first 15

        for img_path in tqdm(images, desc=f"Testing {category_folder.name}"):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                predicted_idx = logits.argmax(-1).item()
                confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()
                
                # Uses your "Pretty Labels" from the model config
                predicted_label = model.config.id2label[predicted_idx]
                
                results.append({
                    "Original_Folder": category_folder.name,
                    "File_Name": img_path.name,
                    "Predicted_Label": predicted_label,
                    "Confidence": f"{confidence:.2%}"
                })
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}")

    # 3. Save everything to CSV on your PC
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n[DONE] CSV saved at: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    batch_test_to_csv(limit=15)
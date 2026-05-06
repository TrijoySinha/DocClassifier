import os
import torch
import csv
from pathlib import Path
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

BASE_DIR = Path(__file__).resolve().parent.parent.parent 
MODEL_PATH = "Trijoy/my-vit-classifier"
VAL_DIR = os.path.join(BASE_DIR, "data", "dataset", "test")
OUTPUT_CSV = os.path.join(BASE_DIR, "classification_results.csv")

# Loading the processor and model
print("Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_image(image):
    """
    Takes a PIL image (from the API) and returns the label
    """
    try:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(-1).item()
        return model.config.id2label[predicted_class_idx]
    except Exception as e:
        return f"Error: {str(e)}"

def run_bulk_classification():
    results_data = []
    if not os.path.exists(VAL_DIR):
        print(f"Directory {VAL_DIR} not found!")
        return

    print(f"Starting classification in: {VAL_DIR}")
    for root, dirs, files in os.walk(VAL_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)
                label = predict_image(full_path)
                actual_folder = os.path.basename(root)
                results_data.append([file, actual_folder, label])
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Actual_Folder", "Predicted_Class"])
        writer.writerows(results_data)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_bulk_classification()
import os
import torch
import csv
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# Path Definition
MODEL_PATH = "Trijoy/my-vit-classifier"
VAL_DIR = "data/dataset/test"
OUTPUT_CSV = "classification_results.csv"

# Loading the processor and model
print("Loading model...")
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

# Prediction Function
def predict_image(image):
    """
    Takes a PIL image and returns the predicted label string
    """
    try:
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        return model.config.id2label[predicted_class_idx]
    
    except Exception as e:
        return f"Error: {str(e)}"
    

# Looping through the Folder
results_data = []

if not os.path.exists(VAL_DIR):
    print(f"Directory {VAL_DIR} not found!")
else:
    print(f"Starting classification in: {VAL_DIR}")

    for root, dirs, files in os.walk(VAL_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, file)

                label = predict_image(full_path)

                actual_folder = os.path.basename(root)

                results_data.append([file, actual_folder, label])
                print(f"File: {file} | Actual: {actual_folder} | Result: {label}")

# Saving to CSV
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Actual_Folder", "Predicted_Class"])
    writer.writerows(results_data)

print(f"\nFinished...Results saved to {OUTPUT_CSV}")
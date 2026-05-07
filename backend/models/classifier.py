import os
import torch
import csv
from pathlib import Path
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from pdf2image import convert_from_path

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
    
def predict_pdf(pdf_path):
    """
    Converts PDF pages to images and classifies each page.
    """
    try:
        # Convert all pages to PIL images (RGB)
        pages = convert_from_path(pdf_path)
        pdf_results = []

        for i, page in enumerate(pages):
            # Reuse your existing predict_image function
            label = predict_image(page)
            pdf_results.append({
                "page": i + 1,
                "label": label
            })
        return pdf_results
    except Exception as e:
        return f"PDF Error: {str(e)}"

def run_bulk_classification():
    results_data = []
    
    if not os.path.exists(VAL_DIR):
        print(f"Directory {VAL_DIR} not found!")
        return

    print(f"Starting classification in: {VAL_DIR}")
    
    for root, dirs, files in os.walk(VAL_DIR):
        for file in files:
            file_lower = file.lower()
            full_path = os.path.join(root, file)
            actual_folder = os.path.basename(root)

            # --- Handle Image Files ---
            if file_lower.endswith((".jpg", ".jpeg", ".png")):
                label = predict_image(full_path)
                results_data.append([file, actual_folder, label])

            # --- Handle PDF Files ---
            elif file_lower.endswith(".pdf"):
                print(f"Processing PDF: {file}")
                try:
                    # Convert PDF pages to PIL images (requires poppler installed)
                    pages = convert_from_path(full_path, fmt="jpeg")
                    for i, page in enumerate(pages):
                        label = predict_image(page)
                        # Append with unique filename for each page
                        row_name = f"{file}_page_{i + 1}"
                        results_data.append([row_name, actual_folder, label])
                except Exception as e:
                    print(f"Failed to process PDF {file}: {e}")
                    results_data.append([file, actual_folder, f"Error: {str(e)}"])

    # Save to CSV
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Actual_Folder", "Predicted_Class"])
        writer.writerows(results_data)
        
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_bulk_classification()
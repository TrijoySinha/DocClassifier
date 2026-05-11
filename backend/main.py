from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = r"C:\Users\KIIT0001\Documents\GeoGo\Task2\backend\models\final_model"

model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

model.eval()

# ---------------- PREDICTION API ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read uploaded image
        image_data = await file.read()

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Process image
        inputs = processor(images=image, return_tensors="pt")

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Convert logits -> probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get highest probability class
        predicted_class_idx = probs.argmax(-1).item()

        # Label + confidence
        label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()

        # Threshold check
        if confidence < 0.90:
            return {
                "label": "Invalid Document",
                "model_prediction": label,
                "confidence": f"{confidence:.2%}"
            }

        # Valid prediction
        return {
            "label": label,
            "confidence": f"{confidence:.2%}"
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
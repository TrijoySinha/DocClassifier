from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL FROM HUGGING FACE ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "Trijoy/task2-document-classifier")

model = ViTForImageClassification.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

model.eval()

# ---------------- PREDICTION API ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=-1)

        predicted_class_idx = probs.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        confidence = probs[0][predicted_class_idx].item()

        if confidence < 0.90:
            return {
                "label": "Invalid Document",
                "model_prediction": label,
                "confidence": f"{confidence:.2%}"
            }

        return {
            "label": label,
            "confidence": f"{confidence:.2%}"
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
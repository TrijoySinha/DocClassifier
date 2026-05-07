from fastapi import APIRouter, UploadFile, File
from services.inference import run_inference
from PIL import Image
from pdf2image import convert_from_bytes
import io

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    
    # Handle PDF
    if file.filename.lower().endswith(".pdf"):
        # Convert PDF bytes to list of PIL images
        images = convert_from_bytes(content)
        results = []
        for i, img in enumerate(images):
            # run_inference already takes a PIL Image based on your code
            label = run_inference(img) 
            results.append({"page": i + 1, "label": label})
        return {"type": "pdf", "data": results}
    
    # Handle standard images (JPG, PNG, etc.)
    else:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        label = run_inference(image)
        return {"type": "image", "data": label}

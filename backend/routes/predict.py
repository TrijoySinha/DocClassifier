from fastapi import APIRouter, UploadFile, File
from services.inference import run_inference
from PIL import Image

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    result = run_inference(image)
    return {"prediction": result}
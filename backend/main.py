from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from paddleocr import PaddleOCR
from PIL import Image
import torch
import io
import os
import re
import tempfile
from dotenv import load_dotenv

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

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

# Loading OCR Model
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    enable_mkldnn=False
)

# OCR Rules
OCR_RULES = {
    "Aadhaar Card": [
        "aadhaar", "uidai", "unique identification", "government of india"
    ],
    "PAN Card": [
        "income tax department", "permanent account number", "pan card"
    ],
    "Voter ID": [
        "election commission", "elector", "voter", "epic"
    ],
    "Driving License": [
        "driving licence", "driving license", "dl no", "licence", "license",
        "transport department"
    ],
    "Passport": [
        "passport", "republic of india", "surname", "given name"
    ],
    "Salary Slip": [
        "salary slip", "payslip", "pay slip", "net pay", "gross salary",
        "basic salary", "deductions", "earnings", "employee id", "provident fund"
    ],
    "Bank Statement": [
        "bank statement", "account statement", "statement of account",
        "transaction", "debit", "credit", "balance", "ifsc", "account number",
        "withdrawal", "deposit"
    ],
    "Application Form": [
        "application form", "applicant", "form no", "signature", "date of birth",
        "address", "declaration", "registration form"
    ],
    "Court Paper": [
        "court", "petitioner", "respondent", "advocate", "judge", "case no",
        "affidavit", "petition"
    ],
    "Mark Sheet": [
        "marksheet", "mark sheet", "statement of marks", "examination",
        "roll number", "grade", "semester", "university", "board"
    ],
    "Experience Letter": [
        "experience letter", "to whom it may concern", "worked with",
        "employment", "designation", "relieving", "organization"
    ],
}

VISUAL_CLASSES = {
    "Aadhaar Card",
    "PAN Card",
    "Voter ID",
    "Driving License",
    "Passport"
}

TEXT_HEAVY_CLASSES = {
    "Salary Slip",
    "Bank Statement",
    "Application Form",
    "Court Paper",
    "Mark Sheet",
    "Experience Letter"
}

# Helpers
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def run_paddle_ocr(image_data: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name

    try:
        result = ocr.predict(temp_path)

        extracted_texts = []

        if result:
            for page in result:
                if page:
                    for line in page:
                        text = line[1][0]
                        extracted_texts.append(text)

        return " ".join(extracted_texts)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_ocr_prediction(text: str):
    text = clean_text(text)

    scores = {}

    for label, keywords in OCR_RULES.items():
        score = 0;

        for keyword in keywords:
            if keyword.lower() in text:
                score += 1

        scores[label] = score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    return best_label, best_score, scores

def get_vit_prediction(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class_idx = probs.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]
    confidence = probs[0][predicted_class_idx].item()

    return label, confidence

def fuse_predictions(vit_label, vit_confidence, ocr_label, ocr_score):
    if vit_label in VISUAL_CLASSES and vit_confidence >= 0.85:
        return vit_label, "ViT"
    
    if ocr_label in TEXT_HEAVY_CLASSES and ocr_score >= 2:
        return ocr_label, "OCR"
    
    if ocr_label in VISUAL_CLASSES and ocr_score >= 2:
        return ocr_label, "OCR"
    
    if vit_confidence >= 0.95:
        return vit_label, "ViT"

    if ocr_score >= 3:
        return ocr_label, "OCR"
    
    return "Invalid Document", "Threshold"


# API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        vit_label, vit_confidence = get_vit_prediction(image)

        extracted_text = run_paddle_ocr(image_data)
        ocr_label, ocr_score, ocr_scores = get_ocr_prediction(extracted_text)

        final_label, source = fuse_predictions(
            vit_label,
            vit_confidence,
            ocr_label,
            ocr_score
        )

        return {
            "label": final_label,
            "confidence": f"{vit_confidence:.2%}",
            "source": source,
            "vit_prediction": vit_label,
            "vit_confidence": f"{vit_confidence:.2%}",
            "ocr_prediction": ocr_label,
            "ocr_score": ocr_score,
            "ocr_text_preview": extracted_text[:300]
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
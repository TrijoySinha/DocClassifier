import os
from config import DATASET_PATH, CACHE_DIR, MODEL_SAVE_PATH
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator,
    EarlyStoppingCallback
)
from torchvision import transforms
from evaluate import load

# 1. Privacy and Storage Shield
# Forcing all temporary data processing to stay on the Pendrive
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# ------------------------------------------- 

# 6. Image Preprocessing and Augmentation
augmentation_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=3, fill=(255, 255, 255)),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

    # Data -> Image -> RGB -> Augmentation -> Tensor -> Stack images -> Labels to Tensors -> Return processed data 
def transform_fn(example_batch):
    pixel_values = [augmentation_pipeline(img.convert("RGB")) for img in example_batch["image"]]
    return {
        "pixel_values": torch.stack(pixel_values),
        "label": torch.tensor(example_batch["label"])
    }

# 2. Custom Label Mapping
custom_label_map = {
    "aadhaar_data": "Aadhaar Card",
    "application_form": "Application Form",
    "bank_statement": "Bank Statement",
    "court_paper": "Court Paper",
    "dl_data": "Driving License",
    "experience_letter": "Experience Letter",
    "marksheets": "Mark Sheet",
    "pan_data": "PAN Card",
    "passport_data": "Passport",
    "payslips": "Salary Slip",
    "voter_data": "Voter ID"
}

def train_model():
    # 3. Load Data
    print(f"Loading dataset from: {DATASET_PATH}...")
    dataset = load_dataset("imagefolder", data_dir=DATASET_PATH)

    # Train/Validation Split
    if "validation" not in dataset:
        print("No validation folder found. Splitting training data 80/20...")
        dataset = dataset["train"].train_test_split(test_size=0.2)
        train_ds = dataset["train"]
        val_ds = dataset["test"]
    else:
        train_ds = dataset["train"]
        val_ds = dataset["validation"]


    # 4. Label Setup
    folder_names = train_ds.features["label"].names
    id2label = {i: custom_label_map.get(name, name) for i, name in enumerate(folder_names)}
    label2id = {label: i for i, label in id2label.items()}

    print(f"Classes found: {list(id2label.values())}")

    # 5. Model & Processor
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    train_ds.set_transform(transform_fn)
    val_ds.set_transform(transform_fn)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_PATH),
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.05,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        per_device_train_batch_size=4,
        dataloader_num_workers=2,           # Speeds up reading from slow USB
        logging_steps=10
    )

    # Metrics
    accuracy_metric = load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Training Start
    print("Starting training... (Junk files are being saved to Pendrive)")
    trainer.train()

    # 9. Saving to PC
    print(f"Saving final model to PC: {MODEL_SAVE_PATH}")
    trainer.save_model(str(MODEL_SAVE_PATH))
    processor.save_pretrained(str(MODEL_SAVE_PATH))
    print("Complete")

if __name__ == "__main__":
    train_model()
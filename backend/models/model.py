import torch
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
import numpy as np
from evaluate import load

# Loading Data 
dataset = load_dataset("imagefolder", data_dir="./dataset")
print(dataset.keys())

# Model
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
labels = dataset["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
    ignore_mismatched_sizes=True
)

# Data Augmentation
augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Handles PIL automatically
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), # This replaces ToImage + ToDtype (it scales to 0.0 - 1.0)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 2. Updated Transform Function
def transform(example_batch):
    pixel_values = [augmentation_pipeline(img.convert("RGB")) for img in example_batch["image"]]
    return {
        "pixel_values": torch.stack(pixel_values),
        "label": torch.tensor(example_batch["label"])
    }

dataset.set_transform(transform)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.05,
    label_smoothing_factor=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    per_device_train_batch_size=4,
    logging_steps=10,
)

accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# Saving
trainer.save_model("final_model")
processor.save_pretrained("final_model")
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataset import load_dataset
from config import *

# ----------------------
# Load tokenizer
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# ----------------------
# Load dataset
# ----------------------
dataset = load_dataset("fraud_text.csv")
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="label"
)

dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# ----------------------
# Load model
# ----------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ----------------------
# Metrics
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ----------------------
# Training Arguments (Transformers v5)
# ----------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True
)

# ----------------------
# Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# ----------------------
# Train
# ----------------------
trainer.train()

# ----------------------
# Save
# ----------------------
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
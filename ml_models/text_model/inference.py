import os
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from config import MAX_LENGTH

# ----------------------
# Setup
# ----------------------
LABELS = ["Legitimate", "Fraudulent"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer + model from saved_model/
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


# ----------------------
# Single-text prediction
# ----------------------
def predict_text(text: str) -> dict:
    """
    Classify a single text string.

    Returns:
        dict with predicted label, confidence, and per-class probabilities.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()

    return {
        "label": LABELS[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "legit_probability": round(probs[0].item(), 4),
        "fraud_probability": round(probs[1].item(), 4),
    }


# ----------------------
# Batch prediction
# ----------------------
def predict_batch(texts: list[str]) -> list[dict]:
    """
    Classify a list of text strings in one forward pass.

    Returns:
        List of result dicts (same format as predict_text).
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    results = []
    for i in range(len(texts)):
        pred_idx = torch.argmax(probs[i]).item()
        results.append({
            "text": texts[i][:80] + ("..." if len(texts[i]) > 80 else ""),
            "label": LABELS[pred_idx],
            "confidence": round(probs[i][pred_idx].item(), 4),
            "legit_probability": round(probs[i][0].item(), 4),
            "fraud_probability": round(probs[i][1].item(), 4),
        })

    return results


# ----------------------
# Demo
# ----------------------
if __name__ == "__main__":
    # --- Single prediction ---
    sample_legit = "Hi team, please find the Q3 report attached. Let me know if you have questions."
    sample_fraud = "URGENT! Your account has been compromised. Click here to verify your identity immediately: http://totallylegit.com/login"

    print("=" * 60)
    print("Single Predictions")
    print("=" * 60)

    result_legit = predict_text(sample_legit)
    result_fraud = predict_text(sample_fraud)

    print(f"\nLegit sample  → {result_legit}")
    print(f"Fraud sample  → {result_fraud}")

    # --- Batch prediction ---
    batch = [sample_legit, sample_fraud]
    print("\n" + "=" * 60)
    print("Batch Prediction")
    print("=" * 60)

    for r in predict_batch(batch):
        print(f"  [{r['label']:>12}] (conf: {r['confidence']:.4f})  {r['text']}")
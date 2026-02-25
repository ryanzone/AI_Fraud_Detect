# 🧠 Text Fraud Detection Model

## 📌 Testing 1 – Transformer-Based Spam/Fraud Classifier

This module implements a Transformer-based fraud detection model using **DistilBERT** fine-tuned on the SMS Spam dataset.

---

## 📊 Dataset

- Dataset: SMS Spam Collection
- Total Samples: 5,572
- Labels:
  - 0 → Legit (Ham)
  - 1 → Fraud (Spam)

---

## 🏗 Model Architecture

- Base Model: `distilbert-base-uncased`
- Fine-tuned for binary classification
- Framework: HuggingFace Transformers (v5.x)
- Backend: PyTorch
- Training Device: CPU

---

## ⚙️ Training Configuration

- Epochs: 2
- Max Length: 128
- Batch Size: 8
- Learning Rate: 2e-5
- Optimizer: AdamW (default HuggingFace)

---

## 📈 Evaluation Results (Testing 1)

| Metric      | Score  |
|------------|--------|
| Accuracy   | 99.01% |
| F1 Score   | 95.94% |
| Precision  | 99.24% |
| Recall     | 92.86% |

---

## 📌 Interpretation

- High precision → Very few false positives
- Strong F1 score → Balanced performance
- Slightly lower recall → Some fraud messages missed (can be improved with threshold tuning)

---

## 🗂 Project Structure
text_model/
│
├── train.py
├── predict.py
├── dataset.py
├── config.py
├── fraud_text.csv
├── saved_model/
└── README.md
# 🧠 Text Fraud Detection Model

## 📌 Transformer-Based Spam/Fraud Classifier (DistilBERT)

This project implements a Transformer-based fraud detection model using **DistilBERT**, fine-tuned on the SMS Spam Collection dataset.

The model is trained using **HuggingFace Transformers (v5.x)** with a **PyTorch backend**, accelerated using **NVIDIA GPU (CUDA)**.

---

## 📊 Dataset

- **Dataset:** SMS Spam Collection  
- **Total Samples:** 5,572  
- **Classes:**
  - `0` → Legitimate (Ham)
  - `1` → Fraudulent (Spam)

A stratified 80/20 train-test split is used to preserve class distribution.

---

## 🏗 Model Architecture

- **Base Model:** `distilbert-base-uncased`
- **Task:** Binary Sequence Classification
- **Framework:** HuggingFace Transformers (v5.x)
- **Backend:** PyTorch
- **Training Device:** NVIDIA GPU (CUDA enabled)
- **Tokenizer:** DistilBERT WordPiece tokenizer
- **Max Sequence Length:** 128

---

## ⚙️ Training Configuration

- **Epochs:** 2  
- **Batch Size:** 8  
- **Learning Rate:** 2e-5  
- **Optimizer:** AdamW (HuggingFace default)  
- **Evaluation Strategy:** Per epoch  
- **Best Model Loaded at End:** Yes  
- **Mixed Precision (fp16):** Enabled  

---

## 🚀 Training Performance

- **Total Training Time:** ~81 seconds (GPU accelerated)  
- **Training Steps/sec:** 13.78  
- **Evaluation Samples/sec:** ~590  

GPU acceleration reduced training time from ~39 minutes (CPU) to ~1.3 minutes.

---

## 📈 Evaluation Results

| Metric      | Epoch 1 | Epoch 2 |
|------------|----------|----------|
| Accuracy   | 99.10%   | 99.01%   |
| F1 Score   | 96.58%   | 96.27%   |
| Precision  | 98.60%   | 97.26%   |
| Recall     | 94.63%   | 95.30%   |
| Eval Loss  | 0.0431   | 0.0461   |

---

## 📌 Interpretation

- High precision → Very few legitimate messages are misclassified as fraud.
- Strong recall → Most fraud messages are correctly detected.
- High F1 score → Balanced performance.
- Slight evaluation loss fluctuation indicates stable training without significant overfitting.

---

## 🧪 Inference

The trained model is stored in:
saved_model/


Prediction is performed using:

The script:
- Loads the fine-tuned model
- Automatically uses GPU if available
- Outputs fraud probability using softmax

Example output:

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


---

## 🔬 Future Improvements

- ROC-AUC / PR-AUC evaluation
- Threshold tuning
- Class-weighted loss
- Cross-validation
- Hyperparameter optimization

---

## 🎯 Status

✔ GPU Accelerated  
✔ Stable Training  
✔ High Classification Performance  
✔ Production-Ready Baseline Model  
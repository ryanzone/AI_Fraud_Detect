import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
model.to(device)   # 🔥 move model to GPU
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")

def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # 🔥 move tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    fraud_prob = probs[0][1].item()

    return fraud_prob


if __name__ == "__main__":
    text = input("Enter text: ")
    score = predict_text(text)
    print(f"Fraud Probability: {score:.4f}")
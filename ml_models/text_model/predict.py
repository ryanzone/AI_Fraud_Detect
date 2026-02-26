import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")

def predict_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    fraud_prob = probs[0][1].item()

    return fraud_prob


if __name__ == "__main__":
    text = input("Enter text: ")
    score = predict_text(text)
    print(f"Fraud Probability: {score:.4f}")
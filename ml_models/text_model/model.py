from transformers import DistilBertForSequenceClassification

def get_model(model_name):
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    return model
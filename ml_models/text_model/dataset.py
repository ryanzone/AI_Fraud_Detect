import pandas as pd
import re
from datasets import Dataset, ClassLabel, Features, Value

def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Merge subject + body if exists
    if "subject" in df.columns and "body" in df.columns:
        df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    df = df[["text", "label"]]
    df["text"] = df["text"].apply(clean_text)

    # Define proper dataset features
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=["legit", "fraud"])
    })

    dataset = Dataset.from_pandas(df, features=features)

    return dataset
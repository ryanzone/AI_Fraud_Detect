import pandas as pd
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Keep required columns
    df = df[["text", "label"]]

    return Dataset.from_pandas(df)
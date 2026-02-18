import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Simple sample dataset
data = {
    "text": [
        "Your bank account is blocked click here",
        "Congratulations you won lottery claim now",
        "Meeting scheduled at 10 AM",
        "Project deadline tomorrow",
        "Verify your account immediately",
        "Important update regarding your password",
        "Let's have lunch tomorrow",
        "Team meeting agenda attached"
    ],
    "label": [1, 1, 0, 0, 1, 1, 0, 0]  # 1 = Fraud, 0 = Safe
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "email_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully!")

from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("email_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Simple URL rule-based checker
def url_check(url):
    suspicious_keywords = ["login", "verify", "secure", "bank", "update"]
    for word in suspicious_keywords:
        if word in url.lower():
            return 1
    return 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    url = data.get("url", "")

    result = {}

    # Text Detection
    if text:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        confidence = model.predict_proba(text_vec)[0][prediction]

        result["text_result"] = "FRAUD" if prediction == 1 else "SAFE"
        result["text_confidence"] = round(float(confidence) * 100, 2)

    # URL Detection
    if url:
        url_pred = url_check(url)
        result["url_result"] = "FRAUD" if url_pred == 1 else "SAFE"

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

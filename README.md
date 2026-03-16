# AI Fraud Detection

A comprehensive system designed to detect AI-generated fraud content, including deepfakes, cloned voices, and LLM-generated text.

## Tech Stack

### 1. Programming
*   **Python:** The core language for AI/ML development (using Jupyter for research).
*   **JavaScript:** For frontend interactivity.

### 2. Frontend
*   **HTML, CSS, Bootstrap:** For a responsive web interface.

### 3. Backend
*   **Flask:** Lightweight Python framework for serving models and application logic.

### 4. Machine Learning
*   **TensorFlow / Keras & PyTorch:** Primary deep learning frameworks.
*   **Scikit-Learn:** Baseline models and evaluation metrics.

### 5. Computer Vision
*   **OpenCV:** Image and video processing for deepfake detection.

### 6. Computer Vision
*   **OpenCV:** Image and video processing for deepfake detection.

### 7. NLP (SMS Detection)
*   **NLTK:** Natural Language Toolkit for text analysis and fraud detection.

### 8. Database
*   **ChromaDB:** Vector database for managing embeddings and retrieval. (See: [Vector DB Guide](backend/db/VECTOR_DB_GUIDE.md))

### 9. Security
*   **JWT Authentication & AES Encryption:** Secure access and data protection.

### 10. Deployment & Tools
*   **Docker:** Containerization.
*   **Git & GitHub:** Version control and collaboration.

## Directory Structure

```text
AI_Fraud_Detect/
├── .env                  # Environment configurations (API Keys)
├── app.py                # Main application entry point
├── requirements.txt      # Global project dependencies
├── backend/              # API and Database layer
│   ├── main.py           # FastAPI application
│   ├── requirements.txt  # Backend specific dependencies
│   └── db/               # Database logic and persistent storage
├── ml_models/            # Machine Learning Modules
│   ├── vision_model/     # Deepfake and image detection
│   ├── text_model/       # LLM-generated text detection
│   └── fusion/           # Multi-modal fusion logic
├── frontend/             # Frontend application files
├── static/               # Static assets (CSS, JS, Images)
├── templates/            # HTML templates for visual interface
└── venv/                 # Python virtual environment
```

# AI Fraud Detection

A comprehensive system designed to detect AI-generated fraud content, including deepfakes, cloned voices, and LLM-generated text.

## Tech Stack

### 1. Programming
*   **Python:** The core language for AI/ML development and media processing.
*   **TypeScript / JavaScript:** For interactive user interfaces.

### 2. Frontend
*   **Next.js (React):** Web dashboard for analyzing uploaded content.
*   **Tailwind CSS:** Modern UI styling.

### 3. Backend
*   **FastAPI:** High-performance asynchronous Python framework for serving ML models over REST APIs.

### 4. Machine Learning
*   **PyTorch:** Primary framework for deep learning models.
*   **Scikit-Learn:** Baseline models and evaluation metrics.

### 5. Computer Vision (Deepfake Detection)
*   **OpenCV:** Image/video frame processing.
*   **XceptionNet / EfficientNet:** Detecting pixel-level facial manipulations.

### 6. Audio Processing (Voice Cloning Detection)
*   **Librosa:** Audio feature extraction.
*   **Wav2Vec 2.0 / ResNet:** Detecting synthetic audio artifacts.

### 7. NLP (SMS & Text Detection)
*   **Hugging Face Transformers:** AI text detection using models like RoBERTa or specialized detectors.

### 8. Database
*   **PostgreSQL:** Securely storing user accounts and scan metadata.
*   **AWS S3 / Local Storage:** Temporary storage for media files during analysis.

### 9. Security
*   **JWT & OAuth2:** API access and authentication.
*   **Rate Limiting:** Protection against bot spam.

### 10. Deployment & Tools
*   **Docker:** Containerizing the stack for consistent deployment.
*   **GitHub Actions:** Continuous Integration.

## Directory Structure
*   `backend/`: FastAPI application and API routes.
*   `frontend/`: Next.js web application.
*   `ml_models/`: ML model weights, data preprocessing, and inference scripts.

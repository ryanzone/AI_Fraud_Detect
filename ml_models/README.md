# Machine Learning Models

This directory contains the scripts for model inference, data processing, and stored weights for:
1. NLP (Text Detection)
2. Computer Vision (Deepfake Detection)
3. Audio Processing (Voice Cloning Detection)

## Directory Structure
```text
ml_models/
├── audio_model/    # Audio Processing (Voice Cloning Detection)
├── fusion/         # Model interaction / fusion scripts
├── text_model/     # NLP (Text Detection)
├── vision_model/   # Computer Vision (Deepfake Detection)
├── api.py          # API serving for ML models
└── README.md
```

### `text_model` (NLP Text Fraud Detection)
This directory handles text-based fraud detection to identify anomalies such as spam or phishing attempts.

* **Core Technology:** Built using a Transformer architecture (`model.py`), specifically HuggingFace's `DistilBertForSequenceClassification` loaded from the pre-trained `distilbert-base-uncased` model.
* **Data Context:** The model is trained as a binary classifier (Legitimate vs. Fraudulent) on several datasets (e.g., `fraud_text.csv`, `phishing.csv`, `spam.csv`). Sequences are tokenized and processed by `dataset.py`.
* **Pipeline:**
  * `train.py` fine-tunes the DistilBERT model. It utilizes mixed precision (fp16) and GPU acceleration to quickly train the classifier, achieving ~99% Accuracy.
  * `predict.py` runs real-world inferences using the resulting `.pth` weights exported to the `saved_model/` folder.

### `vision_model` (Computer Vision Deepfake Detection)
This folder contains the computer vision pipeline, designed for image-based binary classification, perfectly suited for deepfake or fraudulent document detection.

* **Core Technology:** Leverages **ResNet50**, a powerful Convolutional Neural Network (CNN) imported from `torchvision.models` (`model.py`).
* **Transfer Learning:** To optimize training, it uses transfer learning. The pre-trained ImageNet layers of ResNet50 are "frozen," and only the final connected layer is modified and trained (`model.fc = nn.Linear(in_features, num_classes)`). This allows for rapid model updates.
* **Data Context:** `dataset.py` utilizes PyTorch's `ImageFolder` utility, meaning the dataset is physically organized into class folders within `Dataset/Train` and `Dataset/Validation`. All input images are automatically resized to `224x224` pixels and normalized to match the ImageNet standard.
* **Execution:** `train.py` contains the training loop utilizing the Adam optimizer over 5 epochs.



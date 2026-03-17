# Implementation Plan: AI Fraud Detection

This plan bridges the gap between the prototype and a production-ready system, distributing work across the 5-member team with specific resources for each.

## Resource Center

| Resource | Official Documentation | Research/Download Link |
| :--- | :--- | :--- |
| **Flask** | [flask.palletsprojects.com](https://flask.palletsprojects.com/) | [Download PDF Docs](https://flask.palletsprojects.com/en/stable/flask-docs.pdf) |
| **Bootstrap** | [getbootstrap.com](https://getbootstrap.com/) | [Download Compiled CSS/JS](https://github.com/twbs/bootstrap/releases/download/v5.3.3/bootstrap-5.3.3-dist.zip) |
| **TensorFlow** | [tensorflow.org](https://www.tensorflow.org/) | [Install Guide](https://www.tensorflow.org/install) |
| **PyTorch** | [pytorch.org](https://pytorch.org/) | [Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) |
| **NLTK** | [nltk.org](https://www.nltk.org/) | [Data Downloads](https://www.nltk.org/data.html) |
| **OpenCV** | [opencv.org](https://docs.opencv.org/4.x/) | [Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html) |
| **Librosa** | [librosa.org](https://librosa.org/doc/latest/index.html) | [Example Notebooks](https://librosa.org/doc/latest/tutorials.html) |
| **Supabase** | [supabase.com/docs](https://supabase.com/docs) | [Python SDK Guide](https://supabase.com/docs/reference/python/introduction) |
| **JWT** | [jwt.io](https://jwt.io/introduction) | [Debugger/Playground](https://jwt.io/) |
| **AES/Security** | [cryptography.io](https://cryptography.io/en/latest/) | [AES Implementation Guide](https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/) |
| **Docker** | [docs.docker.com](https://docs.docker.com/) | [Download Docker Desktop](https://www.docker.com/products/docker-desktop/) |

---

## Proposed Changes

### 1. **Harshitha** (Backend & Project Lead)
- **Resources:** [Flask Docs](https://flask.palletsprojects.com/), [JWT.io](https://jwt.io/), [Cryptography (AES)](https://cryptography.io/en/latest/)
- Coordinate project milestones and technical architecture.
- Refactor [app.py](file:///d:/programs/AI_Fraud_Detect/app.py) and implement core security (JWT/AES).

---

### 2. **Rupa** (Frontend Design)
- **Resources:** [Bootstrap Docs](https://getbootstrap.com/docs/5.3/getting-started/introduction/), [Bootstrap Examples](https://getbootstrap.com/docs/5.3/examples/)
- Design the user interface using HTML, CSS, and Bootstrap.

---

### 3. **Ryan** (ML Integration)
- **Resources:** [TensorFlow](https://www.tensorflow.org/api_docs), [PyTorch](https://pytorch.org/docs/stable/index.html), [NLTK](https://www.nltk.org/), [OpenCV](https://docs.opencv.org/), [Librosa](https://librosa.org/doc/latest/)
- Implement detection logic using TensorFlow/PyTorch.
- Integrate NLTK for text analysis and Librosa/OpenCV for media analysis.

#### [NEW] [nlp_engine.py](file:///d:/programs/AI_Fraud_Detect/ml_models/nlp_engine.py)
- NLTK processing.

#### [NEW] [vision_engine.py](file:///d:/programs/AI_Fraud_Detect/ml_models/vision_engine.py)
- OpenCV frame analysis.

---

### [Database & Vectors] - **Shivani**
- **Resources:** [Supabase Docs](https://supabase.com/docs)

#### [NEW] [supabase_client.py](file:///d:/programs/AI_Fraud_Detect/backend/db/supabase_client.py)
- Vector database configuration.

---

### [Deployment & Orchestration] - **Aryan**
- **Resources:** [Docker Docs](https://docs.docker.com/), [Compose V2](https://docs.docker.com/compose/)

#### [NEW] [Dockerfile](file:///d:/programs/AI_Fraud_Detect/Dockerfile)
- Flask + ML container.

#### [NEW] [docker-compose.yml](file:///d:/programs/AI_Fraud_Detect/docker-compose.yml)
- Orchestration.

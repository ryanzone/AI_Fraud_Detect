# Vector Database Guide: Local vs. Cloud

This document explains the architecture and decision-making for the Vector Database (ChromaDB) used in the **AI Fraud Detection** project.

---

## 🏗️ Architecture Overview

The system uses **ChromaDB** to store and search for fraud patterns (SMS, Emails, Voice transcripts). It supports two modes of operation: **Local Persistent** and **Chroma Cloud**.

### 1. Chroma Local (Persistent)
Stores the vector data directly on your hard drive in a folder (e.g., `./chroma_data`).

| Feature | Description |
| :--- | :--- |
| **Privacy** | Data never leaves your machine. Best for sensitive PII. |
| **Latency** | Near-zero. No network requests required for searching. |
| **Cost** | 100% Free. |
| **Portability** | Harder to share across different team members or servers. |

### 2. Chroma Cloud
Uses a managed database hosted by Chroma.

| Feature | Description |
| :--- | :--- |
| **Hybrid Search** | Optimized for combined Dense (Meaning) + Sparse (Keyword) indexing. |
| **Scalability** | Handles millions of entries without slowing down your PC. |
| **Management** | No need to manage local files or disk space. |
| **Cost** | Usage-based (Free tier available). |

---

## 🧠 The Search "Brains": Dense vs. Sparse

Regardless of Local or Cloud, we use a **Hybrid Search** approach to detect fraud:

1.  **Dense Search (The "Meaning" Brain):**
    *   **Model:** `Qwen-0.6B`
    *   **Goal:** Understands intent. If a fraudster changes the wording of a "Bank Alert" scam, this will still find it because the *meaning* is the same.
2.  **Sparse Search (The "Keyword" Brain):**
    *   **Model:** `Splade`
    *   **Goal:** Finds exact matches. If a fraudster uses a specific bit.ly link or a unique code, this finds it instantly.

---

## 🛠️ How to Switch

You can toggle between modes in `backend/db/chroma_utils.py` by changing the Client type.

### To use Cloud (Current):
```python
self.client = chromadb.CloudClient(
    tenant=TENANT_ID,
    database=DB_NAME,
    api_key=API_KEY
)
```

### To use Local (For privacy/speed):
```python
self.client = chromadb.PersistentClient(path="./backend/db/local_chroma_db")
```

---

## 🚦 Recommendation Matrix

| Scenario | Recommend |
| :--- | :--- |
| **Development & Testing** | **Local.** It's faster to iterate and you won't hit API limits. |
| **Handling Real User Data** | **Local.** Keeps you compliant with privacy laws (GDPR/CCPA). |
| **Production Launch** | **Cloud.** Ensures the database is always available from any server. |
| **Massive Datasets** | **Cloud.** Offloads the heavy compute and RAM usage to the cloud servers. |

---

## 💡 Using Both (Hybrid Setup)
If you decide to use both, the recommended workflow is:
1.  **Local** acts as a cache for "Recent Fraud Trends."
2.  **Cloud** acts as the "Master Archive" of every fraud pattern ever seen.

#  NexusMind — Unified Offline Multimodal RAG System

> **A fully offline AI assistant** that unifies Text, Image, and Audio understanding through Retrieval-Augmented Generation (RAG).
> Built for enterprise environments needing **data privacy**, **citation transparency**, and **GPU-optimized performance** — all without internet.

---

##  Overview

**NexusMind** is a **locally hosted AI system** that performs intelligent retrieval and reasoning over multiple content types — including **PDFs, DOCX files, images, and audio** — using a combination of **FAISS**, **SentenceTransformer**, and **Llama.cpp**.

It provides:

*  **100% Offline Functionality** (no APIs, no cloud)
*  **Multimodal Understanding** (Text, OCR, Speech)
*  **Citation Transparency** (source-linked answers)
*  **Hardware Adaptive Execution** (CPU/GPU auto-detect)
*  **Single-Page UI** — *All controls visible upfront; zero abstraction.*
*  **Adaptive Model Loading:** Detects hardware to choose between Llama-1B, 3B, or 8B models automatically.
*  **Transparent Results:** Every answer is citation-linked to its source.
*  **Future Ready:** Upcoming support for **video input (frame + audio extraction)**.

---

##  Demonstration Video

 Watch full working demo on YouTube: *https://youtu.be/brJ8DDGZnLM*

---

## 🧱 System Architecture (Summary)

```
User Input
   ↓
Data Ingestion (PDF / DOCX / Image / Audio)
   ↓
Preprocessing & Normalization
   ↓
Embedding Generation (SentenceTransformer)
   ↓
Vector Indexing (FAISS)
   ↓
Query & Context Retrieval (Top-K Matching)
   ↓
Local Inference (Llama.cpp)
   ↓
Output + Citations (Flask UI)
```

---

##  Project Structure

```
NexusMind/
│
├── app.py                 # Flask backend server
├── ingest.py              # Data extraction (PDF, DOCX, OCR, Audio)
├── embed_index.py         # Embedding generation & FAISS indexing
├── query_rag.py           # RAG query + response pipeline
├── hardware_check.py      # System hardware detection
├── templates/             # Frontend HTML files
├── static/                # CSS / JS / Icons / Animations
├── models/                # Local Llama models (1B / 3B / 8B)
├── uploads/               # Uploaded files
├── faiss_index.bin        # Vector index storage
├── metadata.pkl           # Metadata for chunk references
└── requirements.txt       # Python dependencies
```

---

## 🖥️ System Requirements

| Component   | Minimum                | Recommended                                       |
| ----------- | ---------------------- | ------------------------------------------------- |
| **CPU**     | Intel i5 10th Gen      | Intel i9-14900HX                                  |
| **GPU**     | GTX 1650 (4GB)         | RTX 4060 / 5060 (8GB)                             |
| **RAM**     | 8 GB                   | 24 GB DDR5                                        |
| **Storage** | 10 GB free             | 1 TB NVMe SSD                                     |
| **OS**      | Windows 10/11 (64-bit) | Windows 11 Pro                                    |
| **Python**  | 3.9 – **3.10 only** ❗  | *Do not use 3.11+* (FAISS-GPU/CPU not compatible) |

---

##  Software Dependencies

Install all dependencies from `requirements.txt` using:

```bash
pip install -r requirements.txt
```

---

##  Running the Project

**Step 1: Initialize the environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3: Start the application**

```bash
python app.py
```

Then open your browser and go to:
 **[http://localhost:5000](http://localhost:5000)**

---
## Model Handling & Downloads

Default model path: `models/llama-3-8b.gguf`  
Modify configuration in `hardware_check.py` based on system specs.

| Model | Description | Type | Download Link |
|--------|--------------|------|---------------|
| **Llama-1B** | Lightweight CPU model for low-end systems | 🧠 CPU-Friendly | [Download (1B GGUF)](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| **Llama-3B** | Balanced model for CPU + mid GPU usage | ⚙️ Balanced | [Download (3B GGUF)](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| **Llama-8B** | High-performance GPU model (used in NexusMind) | ⚡ GPU Optimized | [Download (8B GGUF)](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) |
| **all-MiniLM-L6-v2** | Text Embedding model (used for FAISS indexing) | 🔍 SentenceTransformer | [Download (Hugging Face)](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF) |
 


> **Important** Use Python **≤ 3.10** for FAISS-GPU support.
>  **Select the version during downloding Q4_K_M.**

---

## Technical Highlights

| Function             | Technology Used                            |
| -------------------- | ------------------------------------------ |
| Data Extraction      | PyMuPDF, python-docx, Pytesseract, Whisper |
| Embedding Generation | SentenceTransformer (MiniLM)               |
| Vector Storage       | FAISS                                      |
| Context Retrieval    | Top-K Similarity Search                    |
| Model Inference      | Llama.cpp (Local)                          |
| Frontend             | HTML, CSS, JavaScript                      |
| Backend              | Flask                                      |
| Monitoring           | psutil, pynvml                             |

---

##  Limitations

* Requires **GPU** for fast inference; CPU mode is slower.
* **Whisper** accuracy depends on audio quality.
* **OCR** accuracy may drop for low-resolution images.
* **FAISS-GPU** currently supports **Python ≤ 3.10** only.

---

## Future Enhancements

*  **Video ingestion support** (frame + speech extraction)
*  **Multi-user collaboration mode**
*  **Hybrid cloud integration** for distributed inference
*  **Local database embedding refresh**
*  **Fine-tuned domain models** (medical, legal, academic)

---

##  Conclusion

NexusMind proves that **AI doesn’t need the internet to be intelligent**.
It delivers **multimodal understanding**, **citation-based transparency**, and **fully local computation** — aligning with mission **ethical and responsible AI**.

>  *Designed for performance. Built for privacy. Inspired by intelligence.*

---

## 👨‍💻 Author

**Abhishek Kumar Vishwakarma**
Department of Computer Science (AI & DS)
Shri Ramswaroop Memorial University, Barabanki
**Guide:** Rohit Sir
**Submission:** IBM Academic Collaboration Program — *10 November 2025*

# 🎓 TDS Virtual Teaching Assistant 🤖

A powerful, fast, and private **Question Answering API** for IIT Madras’ *Tools in Data Science* (TDS) course — built with open-source models, vector search, and Gemini-powered RAG. Supports both text and image-based queries.

> 🚀 **Live Demo**: [https://tdsvirtualta23f1002471-production.up.railway.app](https://tdsvirtualta23f1002471-production.up.railway.app)  
> 🔗 **Access API**: [https://tdsvirtualta23f1002471-production.up.railway.app/qa](https://tdsvirtualta23f1002471-production.up.railway.app/qa)

---

## 📦 Features

- 🔍 Retrieval-Augmented Generation (RAG) using top-10 relevant chunks
- 🧠 Gemini-powered answer generation from extracted context
- 🖼️ Accepts **base64-encoded images** to enhance question context
- 📘 Clean, relevant source references for traceability
- ⚡ Fast and optimized for deployment on Hugging Face or Railway
- 🔒 Secure – uses `.env` secrets and private proxy-based embedding API

---


# 🛠️ TDS Virtual TA: Preprocessing Pipeline

This document outlines the **step-by-step data preparation pipeline** required to run the TDS Virtual Teaching Assistant. Before launching the API, you must generate the embedding vectors from raw course content using the following three scripts.

graph TD
    A[🗂️ Raw Discourse Markdown Files] --> B[🔗 qa_combiner.py<br/>Combine Q&A Pairs]
    B --> C[✂️ chunker_qa_pairs.py<br/>Chunk Long Answers]
    C --> D[🔮 embedding_qa_style.py<br/>Generate Embeddings]
    D --> E[📦 chunk_embeddings_chunkedqa.npz]

    E --> F[🚀 FastAPI App (main.py)]
    F --> G[🧠 Gemini + RAG Response]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#cfc,stroke:#333,stroke-width:2px
    style F fill:#fc9,stroke:#333,stroke-width:2px
    style G fill:#ffc,stroke:#333,stroke-width:2px

## 📡 API Usage

### 🔹 POST `/qa`

Submit a question (optionally with a base64 image) and receive a concise, referenced answer from the course materials.

**Endpoint:**  
```bash
https://tdsvirtualta23f1002471-production.up.railway.app/qa

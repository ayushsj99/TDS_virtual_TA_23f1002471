# 🎓 TDS Virtual Teaching Assistant 🤖

A powerful, fast, and private **Question Answering API** for IIT Madras’ *Tools in Data Science* (TDS) course — built with open-source models, vector search, and Gemini-powered RAG. Supports both text and image-based queries.

> 🚀 **Live Demo**: [https://tdsvirtualta23f1002471-production.up.railway.app](https://tdsvirtualta23f1002471-production.up.railway.app)  
> 🔗 **Access API**: [https://tdsvirtualta23f1002471-production.up.railway.app/qa](https://tdsvirtualta23f1002471-production.up.railway.app/qa)

---

## 📦 Features

- 🔍 Retrieval-Augmented Generation (RAG) using top-10 relevant chunks
- 🖼️ Accepts **base64-encoded images** to enhance question context
- 📘 Clean, relevant source references for traceability
- ⚡ Fast and optimized for deployment on Hugging Face or Railway
- 🔒 Secure – uses `.env` secrets and private proxy-based embedding API

---


# 🛠️ TDS Virtual TA: Preprocessing Pipeline

This document outlines the **step-by-step data preparation pipeline** required to run the TDS Virtual Teaching Assistant. Before launching the API, you must generate the embedding vectors from raw course content using the following three scripts.

**📊 RAG Embedding Pipeline**

1. 🗂️ Raw Markdown Files  
   ⬇  
2. 🔗 `qa_combiner.py` – Combine Q&A  
   ⬇  
3. ✂️ `chunker_qa_pairs.py` – Chunk Long Answers  
   ⬇  
4. 🔮 `embedding_qa_style.py` – Generate Embeddings  
   ⬇  
5. 📦 `chunk_embeddings_chunkedqa.npz`  


## 📡 API Usage

### 🔹 POST `/qa`

Submit a question (optionally with a base64 image) and receive a concise, referenced answer from the course materials.

**Endpoint:**  
```bash
https://tdsvirtualta23f1002471-production.up.railway.app/qa

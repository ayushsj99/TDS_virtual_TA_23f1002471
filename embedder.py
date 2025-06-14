import numpy as np
import json
from tqdm import tqdm
import os
import requests
from dotenv import load_dotenv

# Load AIPROXY_TOKEN from .env
load_dotenv()
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
    "Content-Type": "application/json"
}

texts, chunk_ids, files, embeddings = [], [], [], []

with open("combined_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading Chunks"):
        chunk = json.loads(line)
        texts.append(chunk["text"])
        chunk_ids.append(chunk["chunk_id"])
        files.append(chunk["file"])

print(f"✅ Loaded {len(texts)} chunks")

# Batch in groups of 1000 tokens or 96 inputs (safe limit)
for i in tqdm(range(0, len(texts), 96), desc="Embedding with AI Proxy"):
    batch = texts[i:i+96]
    payload = {
        "model": "text-embedding-3-small",
        "input": batch
    }

    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload)
        res.raise_for_status()
        data = res.json()

        for item in data["data"]:
            embeddings.append(item["embedding"])

    except Exception as e:
        print(f"❌ Failed on batch {i}: {e}")
        for _ in batch:
            embeddings.append([0.0] * 1536)  # fallback

# Save embeddings
np.savez_compressed(
    "chunk_embeddings_openai.npz",
    embeddings=np.array(embeddings, dtype=np.float32),
    chunk_ids=np.array(chunk_ids),
    files=np.array(files),
    texts=np.array(texts)
)

print(f"\n✅ Saved {len(embeddings)} embeddings to chunk_embeddings.npz")


# import google.generativeai as genai
# import numpy as np
# import json
# from tqdm import tqdm
# import os
# from dotenv import load_dotenv

# # Load environment variables (optional)
# load_dotenv()

# # Configure Gemini
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # Load your chunks
# texts, chunk_ids, files, embeddings = [], [], [], []

# with open("combined_chunks.jsonl", "r", encoding="utf-8") as f:
#     for line in tqdm(f, desc="Loading Chunks"):
#         chunk = json.loads(line)
#         texts.append(chunk["text"])
#         chunk_ids.append(chunk["chunk_id"])
#         files.append(chunk["file"])

# # Get embedding model
# embedding_model = genai.embed_content

# # Generate embeddings using textembedding-gecko model
# for text in tqdm(texts, desc="Embedding with Gemini"):
#     try:
#         response = embedding_model(
#             model="models/embedding-001",  # "textembedding-gecko"
#             content=text,
#             task_type="retrieval_document",
#         )
#         embeddings.append(response["embedding"])
#     except Exception as e:
#         print(f"❌ Failed to embed a chunk: {e}")
#         embeddings.append([0.0] * 768)  # fallback to dummy vector

# # Save to NPZ
# np.savez_compressed(
#     "chunk_embeddings.npz",
#     embeddings=np.array(embeddings, dtype=np.float32),
#     chunk_ids=np.array(chunk_ids),
#     files=np.array(files),
#     texts=np.array(texts)
# )

# print(f"\n✅ Saved {len(embeddings)} Gemini embeddings to chunk_embeddings.npz")


# from sentence_transformers import SentenceTransformer
# import numpy as np
# import json
# from tqdm import tqdm

# model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# texts, chunk_ids, files, embeddings = [], [], [], []

# with open("combined_chunks.jsonl", "r", encoding="utf-8") as f:
#     for line in tqdm(f, desc="Loading Chunks"):
#         chunk = json.loads(line)
#         texts.append(chunk["text"])
#         chunk_ids.append(chunk["chunk_id"])
#         files.append(chunk["file"])

# # Batch embedding with progress
# embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# # Save compressed NPZ
# np.savez_compressed(
#     "chunk_embeddings.npz",
#     embeddings=np.array(embeddings, dtype=np.float32),
#     chunk_ids=np.array(chunk_ids),
#     files=np.array(files),
#     texts=np.array(texts)
# )

# print(f"\n✅ Saved {len(embeddings)} embeddings to chunk_embeddings.npz")

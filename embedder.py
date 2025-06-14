from sentence_transformers import SentenceTransformer
import numpy as np
import json
from tqdm import tqdm

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

texts, chunk_ids, files, embeddings = [], [], [], []

with open("combined_chunks.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading Chunks"):
        chunk = json.loads(line)
        texts.append(chunk["text"])
        chunk_ids.append(chunk["chunk_id"])
        files.append(chunk["file"])

# Batch embedding with progress
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

# Save compressed NPZ
np.savez_compressed(
    "chunk_embeddings.npz",
    embeddings=np.array(embeddings, dtype=np.float32),
    chunk_ids=np.array(chunk_ids),
    files=np.array(files),
    texts=np.array(texts)
)

print(f"\n✅ Saved {len(embeddings)} embeddings to chunk_embeddings.npz")

import os
import json
import requests
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load API key
load_dotenv()

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
    "Content-Type": "application/json"
}

# Paths
INPUT_FILE = "chunked_qa_pairs.jsonl"
OUTPUT_FILE = "chunk_embeddings_chunkedqa.npz"

# Storage lists
texts, sources, files, ids, answer_urls, embeddings = [], [], [], [], [], []

# Load data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(tqdm(f, desc="🔄 Loading Chunks")):
        item = json.loads(line)

        if item["source"] == "discourse":
            question = item.get("question", "").strip()
            answer = (item.get("answer") or "").strip()
            content = f"Q: {question}\nA: {answer}"
            file_url = item.get("question_url", "")
            answer_url = item.get("answer_url", "")
            item_id = f"{item.get('question_id', 'unknown')}_{idx}"

        elif item["source"] == "markdown":
            content = item.get("chunk_text", "").strip()
            file_url = item.get("original_url", item.get("source_file", "unknown.md"))
            answer_url = item.get("original_url", "")
            item_id = f"md_{idx}"

        else:
            continue  # skip unknown type

        texts.append(content)
        sources.append(item["source"])
        files.append(file_url)
        ids.append(item_id)
        answer_urls.append(answer_url)

print(f"✅ Loaded {len(texts)} chunks for embedding")

# Batch Embedding
BATCH_SIZE = 96
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="📡 Embedding Chunks"):
    batch = texts[i:i + BATCH_SIZE]
    payload = {
        "model": "text-embedding-3-small",
        "input": batch
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        for item in data["data"]:
            embeddings.append(item["embedding"])
    except Exception as e:
        print(f"❌ Failed batch {i}-{i + BATCH_SIZE}: {e}")
        for _ in batch:
            embeddings.append([0.0] * 1536)  # fallback blank embedding

# Save data
np.savez_compressed(
    OUTPUT_FILE,
    embeddings=np.array(embeddings, dtype=np.float32),
    texts=np.array(texts),
    ids=np.array(ids),
    sources=np.array(sources),
    files=np.array(files),
    answer_urls=np.array(answer_urls)
)

print(f"\n✅ Saved {len(embeddings)} embeddings to {OUTPUT_FILE}")

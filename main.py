from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from google import genai
import os
from PIL import Image
from io import BytesIO
import uvicorn
import json
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved embeddings and metadata
data = np.load("chunk_embeddings_chunkedqa.npz", allow_pickle=True)
embeddings = data["embeddings"]
chunk_ids = data["ids"]
files = data["files"]
texts = data["texts"]
answer_urls = data["answer_urls"]  # <-- used for references

# Request body schema
class QARequest(BaseModel):
    question: str
    image: str | None = None  # base64 image optional


# ---------- Embedding with AI Proxy ----------
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

def embed_with_aiproxy(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": [text]
    }

    res = requests.post(AIPROXY_URL, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]


# ---------- Gemini Image Description ----------
def describe_image(image_base64: str) -> str:
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_bytes))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[img, "Summarize the key visual details of this image in 1–4 sentences. Focus on text, objects, layout."]
    )
    return response.text.strip()


# ---------- Gemini Answer Generation ----------
def ask_gemini(question: str, contexts: list[dict]) -> dict:
    context_text = "\n\n".join([
        f"[{i+1}] {item['text']}\nURL: {item['url']}"
        for i, item in enumerate(contexts)
    ])

    prompt = f"""
You are an expert teaching assistant. Using ONLY the information from the context below, answer the question by going through the material step by step and by reasoning and logic, answer as clearly, concisely, and accurately as possible. Do not make assumptions.

Check if the content has the accurate answer, If the answer is not in the context, reply: "The answer is not available in the provided materials."

Always include direct 1 to 2 most relevant links to relevant sections in the source material using the provided URLs.

Question:
{question}

Context:
{context_text}

Respond strictly in the following JSON format:
enter all links in the links array, and use the text as the link text.
and entry the answer in the answer field.

{{
  "answer": "...",
  "links": [
    {{ "url": "https://...", "text": "..." }},
    ...
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[prompt]
    )
    raw_text = response.text.strip()

    # Clean any ```json formatting
    clean_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except Exception:
        pass

    return {"answer": raw_text, "links": []}


# ---------- QA Route ----------
@app.post("/qa")
def get_answer(request: QARequest):
    try:
        # Step 1: Enhance with image description if any
        full_question = request.question
        if request.image:
            image_description = describe_image(request.image)
            full_question += f"\n\nImage Description: {image_description}"

        # Step 2: Embed question
        question_embed = np.array(embed_with_aiproxy(full_question)).reshape(1, -1)

        # Step 3: Cosine similarity
        dot_products = embeddings @ question_embed.T
        norms_chunks = np.linalg.norm(embeddings, axis=1)
        norm_query = np.linalg.norm(question_embed)
        cosine_similarities = (dot_products.flatten()) / (norms_chunks * norm_query + 1e-10)

        # Step 4: Retrieve top 10
        top_indices = np.argsort(cosine_similarities)[-10:][::-1]
        top_chunks = [
            {
                "text": texts[i],
                "url": answer_urls[i] or files[i] or "https://placeholder.url"
            }
            for i in top_indices
        ]
        # print(f"Question: {request.question}")
        # print(f"Top chunks: {top_chunks}")
        # print(f"Top indices: {top_indices}")
        # print(f"Cosine similarities: {cosine_similarities[top_indices]}")
        

        # Step 5: Ask Gemini
        result = ask_gemini(request.question, top_chunks)
        return result

    except Exception as e:
        return {"error": str(e)}


# ---------- Run Locally ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

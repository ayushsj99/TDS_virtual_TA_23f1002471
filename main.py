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

# Configure Gemini
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

# Load saved embeddings
data = np.load("chunk_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
chunk_ids = data["chunk_ids"]
files = data["files"]
texts = data["texts"]

# Request schema
class QARequest(BaseModel):
    question: str
    image_base64: str | None = None

# AIPROXY config
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Use AI Proxy to embed question
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

# Describe image using Gemini
def describe_image(image_base64: str) -> str:
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_bytes))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[img, "Describe this image in detail for question answering."]
    )
    return response.text.strip()

# Ask Gemini for final answer
def ask_gemini(question: str, contexts: list[str]) -> dict:
    prompt = f"""
You are an expert teaching assistant. Using ONLY the information from the context below, answer the question as clearly, concisely, and accurately as possible. Do not make assumptions or hallucinate.

If the answer is not in the context, reply: "The answer is not available in the provided materials."

Whenever possible, include direct links to relevant sections in the source material.

Question:
{question}

Context:
{chr(10).join(contexts)}

Respond strictly in the following JSON format:

{{
  "answer": "...",
  "links": [
    {{ "url": "https://...", "text": "Click here" }},
    ...
  ]
}}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[prompt]
    )
    raw_text = response.text.strip()

    # Remove triple backtick JSON blocks if any
    clean_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except Exception:
        pass

    return {"answer": raw_text, "links": []}


# FastAPI route
@app.post("/qa")
def get_answer(request: QARequest):
    try:
        # Step 1: Enhance with image description
        full_question = request.question
        if request.image_base64:
            image_description = describe_image(request.image_base64)
            full_question += f"\n\nImage Description: {image_description}"

        # Step 2: Embed with AI Proxy
        question_embed = np.array(embed_with_aiproxy(full_question)).reshape(1, -1)

        # Step 3: Cosine similarity using NumPy
        dot_products = embeddings @ question_embed.T
        norms_chunks = np.linalg.norm(embeddings, axis=1)
        norm_query = np.linalg.norm(question_embed)
        cosine_similarities = (dot_products.flatten()) / (norms_chunks * norm_query + 1e-10)

        # Step 4: Get top 10 most similar chunks
        top_indices = np.argsort(cosine_similarities)[-10:][::-1]
        top_chunks = [texts[i] for i in top_indices]

        # Step 5: Ask Gemini and return result
        result = ask_gemini(request.question, top_chunks)
        return result

    except Exception as e:
        return {"error": str(e)}


# Uvicorn entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)



# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import base64
# import numpy as np
# from google import genai
# import os
# from PIL import Image
# from io import BytesIO
# from sklearn.metrics.pairwise import cosine_similarity
# import uvicorn
# import json
# import re
# import requests

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Configure Gemini API
# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# # FastAPI setup
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# data = np.load("chunk_embeddings.npz", allow_pickle=True)
# embeddings = data["embeddings"]
# chunk_ids = data["chunk_ids"]
# files = data["files"]
# texts = data["texts"]


# # Request body
# class QARequest(BaseModel):
#     question: str
#     image_base64: str | None = None


# AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
# AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# def embed_with_aiproxy(text: str) -> list[float]:
#     headers = {
#         "Authorization": f"Bearer {AIPROXY_TOKEN}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "model": "text-embedding-3-small",
#         "input": [text]
#     }

#     res = requests.post(AIPROXY_URL, headers=headers, json=payload)
#     res.raise_for_status()
#     return res.json()["data"][0]["embedding"]


# # Helper: Get image description
# def describe_image(image_base64: str) -> str:
#     img_bytes = base64.b64decode(image_base64)
#     img = Image.open(BytesIO(img_bytes))
#     response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[img, "Describe this image in detail for question answering."]
# )
#     return response.text.strip()

# # Helper: Query Gemini for answer

# def ask_gemini(question: str, contexts: list[str]) -> dict:
#     response = client.models.generate_content(
#                                            model="gemini-2.5-flash-preview-05-20"
#     content = [f"""
# Answer the following question using the given context chunks. Add links to your sources.

# Question: {question}

# Context:
# {chr(10).join(contexts)}

# Format your answer as:
# {{
#   "answer": "...",
#   "links": [
#     {{ "url": "...", "text": "..." }},
#     ...
#   ]
# }}"""])
#     raw_text = response.text.strip()

#     # Strip out markdown formatting if present
#     clean_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

#     try:
#         parsed = json.loads(clean_text)
#         if isinstance(parsed, dict) and "answer" in parsed:
#             return parsed
#     except Exception:
#         pass

#     return {"answer": raw_text, "links": []}

    
    

# @app.post("/qa")
# def get_answer(request: QARequest):
#     try:
#         # Step 1: Enhance question with image
#         full_question = request.question
#         if request.image_base64:
#             image_description = describe_image(request.image_base64)
#             full_question += f"\n\nImage Description: {image_description}"

#         # Step 2: Embed question
#         question_embed = np.array(embed_with_aiproxy(full_question)).reshape(1, -1)

#         # Step 3: Compute similarities
#         sims = cosine_similarity(question_embed, embeddings)[0]
#         top_indices = sims.argsort()[-10:][::-1]
#         top_chunks = [texts[i] for i in top_indices]

#         # Step 4: Ask Gemini
#         result = ask_gemini(request.question, top_chunks)
#         return result

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Railway uses dynamic port
#     uvicorn.run("main:app", host="0.0.0.0", port=port)

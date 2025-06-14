from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import json
import re

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and chunks
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
data = np.load("chunk_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]
chunk_ids = data["chunk_ids"]
files = data["files"]
texts = data["texts"]

# Request body
class QARequest(BaseModel):
    question: str
    image_base64: str | None = None

# Helper: Get image description
def describe_image(image_base64: str) -> str:
    img_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(img_bytes))
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content([
        img,
        "Describe this image in detail for question answering."
    ])
    return response.text.strip()

# Helper: Query Gemini for answer

def ask_gemini(question: str, contexts: list[str]) -> dict:
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    prompt = f"""
Answer the following question using the given context chunks. Add links to your sources.

Question: {question}

Context:
{chr(10).join(contexts)}

Format your answer as:
{{
  "answer": "...",
  "links": [
    {{ "url": "...", "text": "..." }},
    ...
  ]
}}
"""
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip out markdown formatting if present
    clean_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except Exception:
        pass

    return {"answer": raw_text, "links": []}

    
    

@app.post("/qa")
def get_answer(request: QARequest):
    try:
        # Step 1: Enhance question with image
        full_question = request.question
        if request.image_base64:
            image_description = describe_image(request.image_base64)
            full_question += f"\n\nImage Description: {image_description}"

        # Step 2: Embed question
        question_embed = embed_model.encode([full_question])[0].reshape(1, -1)

        # Step 3: Compute similarities
        sims = cosine_similarity(question_embed, embeddings)[0]
        top_indices = sims.argsort()[-10:][::-1]
        top_chunks = [texts[i] for i in top_indices]

        # Step 4: Ask Gemini
        result = ask_gemini(request.question, top_chunks)
        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

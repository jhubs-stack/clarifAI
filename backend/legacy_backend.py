import os
import json
import pickle
import numpy as np
import faiss

# Load FAQs from JSON
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]

# Generate a mock embedding (random but deterministic)
def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536).astype("float32")

# Build and save the FAISS index
def build_index():
    print("🔄 Generating mock embeddings...")
    embeddings = [get_embedding(q) for q in questions]
    embeddings_np = np.array(embeddings).astype("float32")

    print("📦 Creating FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings_np[0]))
    index.add(embeddings_np)

    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump((index, faqs), f)

    print("✅ Mock index built and saved to vectorstore/faiss_index.pkl")

# Load the saved index and FAQ data
def load_index():
    index_path = "vectorstore/faiss_index.pkl"

    # If missing, auto-build
    if not os.path.exists(index_path):
        os.makedirs("vectorstore", exist_ok=True)
        build_index()

    with open(index_path, "rb") as f:
        index, faqs = pickle.load(f)

    return index, faqs

# Search the closest FAQ match using FAISS
def search_faq(query, index, faqs, k=1):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)
    best_match_idx = indices[0][0]
    return faqs[best_match_idx]

# Generate clean, markdown-ready response (ClarifAI label handled in app.py)
def generate_response(user_question, matched_faq):
    return matched_faq["answer"]

# CLI testing loop (optional)
if __name__ == "__main__":
    index, faqs = load_index()
    print("🤖 ClarifAI is ready. Type a question or 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        match = search_faq(user_input, index, faqs)
        response = generate_response(user_input, match)
        print(f"\nClarifAI:\n{response}")
        print(f"📌 Matched FAQ: {match['question']}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

# Load FAISS index once at startup
index, faqs = load_index()

@app.post("/chat")
def chat(msg: Message):
    user_input = msg.message
    match = search_faq(user_input, index, faqs)
    response = generate_response(user_input, match)
    return {
        "reply": response,
        "matched_question": match["question"]
    }

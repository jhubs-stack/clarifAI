import os
import json
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAQs from JSON
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

# Extract the questions
questions = [faq["question"] for faq in faqs]

# Generate an embedding for each question
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Lightweight and fast
    )
    return response.data[0].embedding

# Build and save the FAISS index
def build_index():
    print("🔄 Generating embeddings...")
    embeddings = [get_embedding(q) for q in questions]
    embeddings_np = np.array(embeddings).astype("float32")

    print("📦 Creating FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings_np[0]))
    index.add(embeddings_np)

    # Save index and FAQs together
    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump((index, faqs), f)

    print("✅ Index built and saved to vectorstore/faiss_index.pkl")

# Run only if this file is executed directly
if __name__ == "__main__":
    build_index()

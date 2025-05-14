import os
import json
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv

# Load environment variables (OpenAI API key)
load_dotenv()

# Load FAQs from JSON
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

# Extract the questions
questions = [faq["question"] for faq in faqs]

# Generate a mock embedding for each question
def get_embedding(text):
    # Return a fixed-size random vector for demonstration purposes
    np.random.seed(hash(text) % (2**32))
    return np.random.rand(1536).astype("float32")

# Build and save the FAISS index
def build_index():
    print("🔄 Generating mock embeddings...")
    embeddings = [get_embedding(q) for q in questions]
    embeddings_np = np.array(embeddings).astype("float32")

    print("📦 Creating FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings_np[0]))
    index.add(embeddings_np)

    # Save index and FAQs together
    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump((index, faqs), f)

    print("✅ Mock index built and saved to vectorstore/faiss_index.pkl")

# Run only if this file is executed directly
if __name__ == "__main__":
    build_index()

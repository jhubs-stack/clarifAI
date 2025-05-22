import faiss
import pickle
import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (to access OpenAI API)
load_dotenv()
client = OpenAI()

# Load FAQ data
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

# Embed questions
def embed(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

print("Embedding questions...")
embeddings = np.array([embed(faq["question"]) for faq in faqs]).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save as tuple (index, faqs)
with open("vectorstore/faiss_index.pkl", "wb") as f:
    pickle.dump((index, faqs), f)

print("✅ FAISS index generated and saved.")

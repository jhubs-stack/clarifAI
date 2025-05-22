import json
import numpy as np
import faiss
from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle

load_dotenv()
client = OpenAI()

# Load FAISS index
def load_index():
    with open("vectorstore/faiss_index.pkl", "rb") as f:
        data = pickle.load(f)
    return data

# Simple search with FAISS
def search_faq(data, query):
    index = data["index"]
    faqs = data["faqs"]
    query_vector = embed_text(query)
    D, I = index.search(np.array([query_vector]).astype("float32"), k=3)
    return [faqs[i] for i in I[0] if i < len(faqs)]

# Embedding helper using OpenAI
def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Generate answer
def generate_response(query, context):
    context_text = "\n".join([f"- {item['question']}: {item['answer']}" for item in context])
    prompt = (
        f"You are Frankie, the helpful shoe store AI.\n"
        f"Customer question: {query}\n\n"
        f"Based on the following FAQ context, provide a concise, friendly response:\n"
        f"{context_text}"
    )
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return completion.choices[0].message.content

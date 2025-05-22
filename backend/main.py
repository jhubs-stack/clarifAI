from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI()

app = FastAPI()

# Allow frontend dev server to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"  # local dev only
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and FAQs from file
def load_index():
    with open("vectorstore/faiss_index.pkl", "rb") as f:
        index, faqs = pickle.load(f)
    return {"index": index, "faqs": faqs}

# Embed the user's query into a vector
def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Search FAISS for top-k similar questions
def search_faq(data, query, k=3):
    index = data["index"]
    faqs = data["faqs"]


    query_vector = embed_text(query)
    query_vector_np = np.array(query_vector, dtype="float32").reshape(1, -1)

    D, I = index.search(query_vector_np, k)
    return [faqs[i] for i in I[0] if i < len(faqs)]

# Generate a chatbot response using OpenAI
def generate_response(query, context):
    context_text = "\n".join([f"- {item['question']}: {item['answer']}" for item in context])
    prompt = (
        f"You are Frankie, the helpful AI assistant for a shoe store.\n"
        f"Customer asked: {query}\n\n"
        f"Based on the following FAQs, provide a friendly, clear, and concise response:\n"
        f"{context_text}"
    )
    print("🧾 Context sent to OpenAI:\n", context_text)
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Load vectorstore at startup
data = load_index()

# Health check route
@app.get("/")
def read_root():
    return {"message": "HappyFeet backend is up!"}

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        query = body.get("message")
        if not query:
            return {"response": "No message provided."}
        context = search_faq(data, query)
        answer = generate_response(query, context)
        return {"response": answer}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}"}

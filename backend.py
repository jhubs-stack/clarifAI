import os
import json
import pickle
import numpy as np
import faiss

# Load FAQ data
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]

# Generate a mock embedding (random vector) for each question
def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**32))  # deterministic random vector
    return np.random.rand(1536).astype("float32")

# Build and save the FAISS index
def build_index():
    print("🔄 Generating mock embeddings...")
    embeddings = [get_embedding(q) for q in questions]
    embeddings_np = np.array(embeddings).astype("float32")

    print("📦 Creating FAISS index...")
    index = faiss.IndexFlatL2(len(embeddings_np[0]))
    index.add(embeddings_np)

    # Save index and FAQs
    with open("vectorstore/faiss_index.pkl", "wb") as f:
        pickle.dump((index, faqs), f)

    print("✅ Mock index built and saved to vectorstore/faiss_index.pkl")

if __name__ == "__main__":
    # Uncomment this if you still want to regenerate the index
    build_index()

    index, faqs = load_index()
    print("🤖 ClarifAI is ready. Ask me a question!")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        match = search_faq(user_input, index, faqs)
        response = generate_response(user_input, match)
        print(response)

# Load the FAISS index and FAQs from disk
def load_index():
    with open("vectorstore/faiss_index.pkl", "rb") as f:
        index, faqs = pickle.load(f)
    return index, faqs

# Search the best-matching FAQ given a user query
def search_faq(query, index, faqs, k=1):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)
    best_match_idx = indices[0][0]
    return faqs[best_match_idx]

# Generate a friendly response based on the best match
def generate_response(user_question, matched_faq):
    return f"""
🤖 ClarifAI says:
{matched_faq['answer']}

📌 (Matched FAQ: {matched_faq['question']})
"""

import os
import json
import pickle
import numpy as np
import faiss

# Load FAQs from JSON
with open("data/faqs.json", "r") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]

# Generate a mock embedding (random vector)
def get_embedding(text):
    np.random.seed(abs(hash(text)) % (2**32))  # deterministic for same input
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

    # If index file doesn't exist, build it
    if not os.path.exists(index_path):
        os.makedirs("vectorstore", exist_ok=True)
        build_index()

    # Load index
    with open(index_path, "rb") as f:
        index, faqs = pickle.load(f)

    return index, faqs

# Search the closest FAQ for a given query
def search_faq(query, index, faqs, k=1):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_vector, k)
    best_match_idx = indices[0][0]
    return faqs[best_match_idx]

# Generate a friendly response from matched FAQ
def generate_response(user_question, matched_faq):
    return f"""
🤖 ClarifAI says:
{matched_faq['answer']}

📌 (Matched FAQ: {matched_faq['question']})
"""

# Run the assistant interactively
if __name__ == "__main__":
    build_index()  # Uncomment if you need to rebuild the index

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

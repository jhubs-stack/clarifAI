import pickle
import faiss

with open("vectorstore/faiss_index.pkl", "rb") as f:
    index, faqs = pickle.load(f)

print("✅ Successfully loaded faiss_index.pkl")

print(f"Type of index: {type(index)}")
print(f"Type of faqs: {type(faqs)}")
print(f"Number of FAQs: {len(faqs)}")

# Check FAISS index dimensions
if isinstance(index, faiss.Index):
    print("FAISS index is valid.")
    print(f"Index dimensions: {index.d}")
else:
    print("❌ Loaded index is NOT a valid FAISS index.")

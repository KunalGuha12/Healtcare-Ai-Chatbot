# ingestion/query_example.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sys
from utils_ingestion import load_metadata

MODEL = "all-MiniLM-L6-v2"

def load_index(index_path):
    return faiss.read_index(index_path)

def embed_query(text, model):
    emb = model.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    return emb

if __name__ == "__main__":
    index_path = "index/sabudh_faiss.index"
    meta_path = "index/sabudh_faiss_metadata.json"
    if len(sys.argv) < 2:
        print("Usage: python query_example.py \"your question here\"")
        sys.exit(1)
    query = sys.argv[1]
    print("Loading model and index (may take a sec)...")
    model = SentenceTransformer(MODEL)
    idx = load_index(index_path)
    emb = embed_query(query, model)
    k = 5
    D, I = idx.search(emb, k)
    metadata = load_metadata(meta_path)
    for rank, doc_id in enumerate(I[0]):
        if doc_id < 0:
            continue
        meta = metadata[doc_id]
        print(f"\nRank {rank+1} - score: {D[0][rank]:.4f}")
        print("Source:", meta.get("source"))
        print("Page:", meta.get("page"))
        print("Chunk id:", meta.get("chunk_id"))
        print("Text snippet:", meta.get("text")[:500].replace("\n"," "))

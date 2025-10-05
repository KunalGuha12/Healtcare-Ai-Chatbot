# ingestion/utils_ingestion.py
import os
import json
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Choose model (downloads first time)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load model once
_model = None
def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def extract_pages_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Return a list of dicts: [{'page': 1, 'text': '...'}, ...]
    Uses PyMuPDF (fitz) for reliable extraction.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append({"page": i + 1, "text": text})
    return pages

def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace and remove weird control chars."""
    if not text:
        return ""
    txt = text.replace("\r", " ")
    txt = txt.replace("\xa0", " ")
    txt = " ".join(txt.split())
    return txt.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Character-based chunker with overlap.
    chunk_size: target characters in a chunk
    overlap: overlapping characters between chunks
    """
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= text_len:
            break
        start += step
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings (numpy float32).
    Returns normalized embeddings for cosine similarity (if using IndexFlatIP).
    """
    model = get_embedding_model()
    # convert_to_numpy=True ensures numpy array
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    # Normalize for cosine similarity when using inner product index
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Build and save a FAISS IndexFlatIP (inner product) index. The embeddings should already be normalized.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via normalized vectors
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

def save_metadata(metadata: List[Dict], path: str):
    """Save metadata list (list of dicts) to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

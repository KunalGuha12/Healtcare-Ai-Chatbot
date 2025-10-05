# ingestion/main_ingestion.py
import os
import argparse
import numpy as np
from utils_ingestion import (
    extract_pages_from_pdf,
    clean_text,
    chunk_text,
    embed_texts,
    build_faiss_index,
    save_metadata,
)

def process_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    For a single PDF: extract text per page, chunk per page,
    create metadata entries and return list of chunk texts + metadata list.
    """
    pages = extract_pages_from_pdf(pdf_path)
    chunks = []
    metadata = []
    global_chunk_id = 0
    for p in pages:
        page_num = p["page"]
        page_text = clean_text(p["text"])
        if not page_text:
            continue
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(page_chunks):
            metadata.append({
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "chunk_id": global_chunk_id,
                "text": ch
            })
            chunks.append(ch)
            global_chunk_id += 1
    return chunks, metadata

def run_ingestion(input_dir, output_dir, index_name="faiss_index", chunk_size=1000, overlap=200):
    """
    Processes all PDFs in input_dir and writes:
      - FAISS index file at output_dir/{index_name}.index
      - metadata at output_dir/{index_name}_metadata.json
    """
    os.makedirs(output_dir, exist_ok=True)
    all_chunks = []
    all_metadata = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, fname)
            print(f"Processing: {pdf_path}")
            chunks, metadata = process_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

    if not all_chunks:
        print("No text chunks found. Make sure PDFs are in the input directory and contain selectable text.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = embed_texts(all_chunks)  # normalized numpy float32

    index_path = os.path.join(output_dir, f"{index_name}.index")
    print("Building FAISS index...")
    build_faiss_index(embeddings, index_path)
    print(f"FAISS index saved to: {index_path}")

    meta_path = os.path.join(output_dir, f"{index_name}_metadata.json")
    print("Saving metadata...")
    save_metadata(all_metadata, meta_path)
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ingestion: PDF -> chunks -> embeddings -> FAISS")
    parser.add_argument("--input_dir", type=str, default="data", help="Folder containing PDF files")
    parser.add_argument("--output_dir", type=str, default="index", help="Where to save index + metadata")
    parser.add_argument("--index_name", type=str, default="sabudh_faiss", help="Name of the FAISS index")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    run_ingestion(args.input_dir, args.output_dir, args.index_name, args.chunk_size, args.overlap)

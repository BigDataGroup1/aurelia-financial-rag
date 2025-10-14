#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split parsed PDF text into smaller overlapping chunks for embedding
"""
import json, os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(parsed_path, chunk_size=800, overlap=200):
    with open(parsed_path) as f:
        pages = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for p in pages:
        for chunk in splitter.split_text(p["text"]):
            chunks.append({
                "page": p["page"],
                "chunk": chunk.strip()
            })
    return chunks

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    chunks = chunk_texts("data/processed/fintbx_parsed.json")
    with open("data/processed/fintbx_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"✅ Created {len(chunks)} chunks saved to data/processed/fintbx_chunks.json")

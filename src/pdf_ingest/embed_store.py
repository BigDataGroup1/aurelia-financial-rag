#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed first 10 pages of fintbx.pdf for testing
Stores in local ChromaDB for verification
"""

import os
import json
import uuid
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Load .env
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")

# Initialize OpenAI + Chroma
client = OpenAI(api_key=OPENAI_KEY)
chroma_client = chromadb.Client(Settings(persist_directory="data/chroma_index_test"))
collection = chroma_client.get_or_create_collection("fintbx_test_chunks")

# =============== Helper Functions ==================

def embed_texts_batch(texts, model="text-embedding-3-large", batch_size=20):
    """Embed texts in small batches for safety."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(texts)//batch_size)+1} ...")
        response = client.embeddings.create(input=batch, model=model)
        embeddings.extend([d.embedding for d in response.data])
    return embeddings

# =============== Main Workflow =====================

if __name__ == "__main__":
    print("🚀 Starting embedding for first 10 pages...")

    # Load chunked data
    with open("data/processed/fintbx_chunks.json") as f:
        all_chunks = json.load(f)

    # Filter chunks where page ≤ 10
    chunks = [c for c in all_chunks if c["page"] <= 10]
    print(f"Found {len(chunks)} chunks from first 10 pages.")

    # Prepare text list
    texts = [c["chunk"] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"page": c["page"]} for c in chunks]

    # Embed
    embeddings = embed_texts_batch(texts)

    # Store
    collection.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=texts)
    print(f"✅ Stored {len(chunks)} embeddings in ChromaDB (data/chroma_index_test)")
    print("🎯 Test run successful! You can now inspect retrieval results.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick retrieval sanity check for embedded fintbx corpus
"""
import chromadb
from chromadb.config import Settings

if __name__ == "__main__":
    # Use the same directory used during embedding
    persist_dir = "data/chroma_index_test"  # or "data/chroma_index" if that’s what you used
    collection_name = "fintbx_test_chunks"  # or "fintbx_chunks" depending on your embed_store.py

    client = chromadb.Client(Settings(persist_directory=persist_dir))

    # If you’re not sure which one exists, list them
    print("Available collections:", client.list_collections())

    # Load the existing collection
    collection = client.get_collection(collection_name)

    query = "What is portfolio optimization?"
    result = collection.query(query_texts=[query], n_results=3)

    print("\n🎯  Query:", query)
    for i, doc in enumerate(result["documents"][0], 1):
        meta = result["metadatas"][0][i - 1]
        print(f"\nResult {i} (page {meta.get('page')}):\n{doc[:300]}...")

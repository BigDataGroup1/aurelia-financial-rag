"""
Lab 1 Step 3: Generate embeddings using text-embedding-3-large
Store in vector database (ChromaDB or Pinecone)
"""
import os
from pathlib import Path
from typing import List, Dict
import json
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large default dimension

# Choose which chunking strategy to embed
# Change this based on your comparison analysis
DEFAULT_STRATEGY = "6_code_aware_1200_200"  # Recommended from analysis


def load_chunks(strategy_name: str) -> List[Dict]:
    """Load chunks from JSON"""
    chunks_file = CHUNKS_DIR / f"{strategy_name}_chunks.json"
    
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"✓ Loaded {len(chunks)} chunks from {strategy_name}")
    return chunks


def generate_embeddings_batch(texts: List[str], client: OpenAI, batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings in batches to handle rate limits
    OpenAI allows up to 3,000 RPM (requests per minute) on standard tier
    """
    all_embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1
    
    logger.info(f"Generating embeddings in {total_batches} batch(es) of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"  ✓ Batch {batch_num}/{total_batches} complete")
            
            # Rate limiting: wait between batches to avoid hitting limits
            if i + batch_size < len(texts):
                time.sleep(0.5)  # 500ms pause between batches
                
        except Exception as e:
            logger.error(f"❌ Error embedding batch {batch_num}: {e}")
            raise
    
    return all_embeddings


def embed_chunks(chunks: List[Dict], strategy_name: str):
    """
    Generate embeddings for all chunks using OpenAI text-embedding-3-large
    """
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set!\n"
            "Set it with: $env:OPENAI_API_KEY='your-key-here'"
        )
    
    client = OpenAI(api_key=api_key)
    
    logger.info(f"\n{'='*70}")
    logger.info("GENERATING EMBEDDINGS")
    logger.info('-'*70)
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    logger.info(f"Total chunks: {len(chunks)}")
    
    # Calculate estimated cost
    total_chars = sum(chunk['char_count'] for chunk in chunks)
    estimated_tokens = total_chars / 4  # Rough estimate: 1 token ≈ 4 chars
    estimated_cost = (estimated_tokens / 1_000_000) * 0.13  # $0.13 per 1M tokens
    
    logger.info(f"Estimated tokens: ~{estimated_tokens:,.0f}")
    logger.info(f"Estimated cost: ~${estimated_cost:.4f}")
    logger.info('-'*70)
    
    # Extract text content
    texts = [chunk['content'] for chunk in chunks]
    
    start_time = time.time()
    
    # Generate embeddings
    embeddings = generate_embeddings_batch(texts, client, batch_size=100)
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    logger.info(f"  Avg time per chunk: {elapsed / len(chunks):.3f}s")
    logger.info(f"  Throughput: {len(chunks) / elapsed:.1f} chunks/sec")
    logger.info("="*70)
    
    # Combine chunks with embeddings
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            'chunk_id': chunk['chunk_id'],
            'content': chunk['content'],
            'metadata': chunk['metadata'],
            'char_count': chunk['char_count'],
            'embedding': embedding,
            'embedding_model': EMBEDDING_MODEL,
            'embedding_dimensions': len(embedding)
        })
    
    # Save embeddings
    output_file = EMBEDDINGS_DIR / f"{strategy_name}_embeddings.json"
    
    logger.info(f"\nSaving embeddings to: {output_file.name}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, indent=2)
    
    logger.info(f"✓ Saved {len(embedded_chunks)} embeddings")
    logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save metadata separately (for quick loading without embeddings)
    metadata_file = EMBEDDINGS_DIR / f"{strategy_name}_embedding_metadata.json"
    metadata = {
        'strategy': strategy_name,
        'model': EMBEDDING_MODEL,
        'dimensions': EMBEDDING_DIMENSIONS,
        'num_chunks': len(embedded_chunks),
        'total_chars': total_chars,
        'generation_time_sec': elapsed,
        'avg_time_per_chunk': elapsed / len(chunks),
        'estimated_tokens': int(estimated_tokens),
        'estimated_cost_usd': round(estimated_cost, 4)
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata: {metadata_file.name}")
    
    return embedded_chunks, metadata


def store_in_chromadb(embedded_chunks: List[Dict], collection_name: str = "fintbx"):
    """
    Store embeddings in ChromaDB (local vector database)
    """
    try:
        import chromadb
    except ImportError:
        logger.warning("\n⚠️  ChromaDB not installed. Run: pip install chromadb")
        logger.warning("   Skipping ChromaDB storage...")
        return False
    
    logger.info(f"\n{'='*70}")
    logger.info("STORING IN CHROMADB")
    logger.info('-'*70)
    
    # Initialize ChromaDB
    chroma_path = PROJECT_ROOT / "data" / "chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Financial Toolbox embeddings"}
    )
    logger.info(f"✓ Created collection: {collection_name}")
    
    # Prepare data for ChromaDB
    ids = [f"chunk_{c['chunk_id']}" for c in embedded_chunks]
    embeddings = [c['embedding'] for c in embedded_chunks]
    documents = [c['content'] for c in embedded_chunks]
    metadatas = [c['metadata'] for c in embedded_chunks]
    
    # Add to collection in batches
    batch_size = 100
    total_batches = (len(embedded_chunks) - 1) // batch_size + 1
    
    logger.info(f"Storing {len(embedded_chunks)} vectors in {total_batches} batch(es)...")
    
    for i in range(0, len(embedded_chunks), batch_size):
        batch_num = i // batch_size + 1
        end_idx = min(i + batch_size, len(embedded_chunks))
        
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        logger.info(f"  ✓ Batch {batch_num}/{total_batches} stored")
    
    logger.info(f"\n✓ Successfully stored {len(embedded_chunks)} embeddings in ChromaDB")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Path: {chroma_path}")
    logger.info("="*70)
    
    return True


def store_in_pinecone(embedded_chunks: List[Dict], index_name: str = "fintbx"):
    """
    Store embeddings in Pinecone (cloud vector database)
    Optional - only runs if PINECONE_API_KEY is set
    """
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        logger.warning("\n⚠️  Pinecone not installed. Run: pip install pinecone-client")
        logger.warning("   Skipping Pinecone storage...")
        return False
    
    logger.info(f"\n{'='*70}")
    logger.info("STORING IN PINECONE")
    logger.info('-'*70)
    
    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning("PINECONE_API_KEY not set. Skipping Pinecone storage.")
        logger.info("To enable: $env:PINECONE_API_KEY='your-key-here'")
        return False
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info(f"✓ Created index: {index_name}")
        time.sleep(10)  # Wait for index to be ready
    else:
        logger.info(f"Using existing index: {index_name}")
    
    # Get index
    index = pc.Index(index_name)
    
    # Prepare vectors for Pinecone
    vectors = []
    for chunk in embedded_chunks:
        # Pinecone has metadata size limits, so truncate content
        metadata = {
            **chunk['metadata'],
            'content_preview': chunk['content'][:1000],  # First 1000 chars only
            'char_count': chunk['char_count'],
            'chunk_id': chunk['chunk_id']
        }
        
        vectors.append({
            'id': f"chunk_{chunk['chunk_id']}",
            'values': chunk['embedding'],
            'metadata': metadata
        })
    
    # Upsert in batches
    batch_size = 100
    total_batches = (len(vectors) - 1) // batch_size + 1
    
    logger.info(f"Upserting {len(vectors)} vectors in {total_batches} batch(es)...")
    
    for i in range(0, len(vectors), batch_size):
        batch_num = i // batch_size + 1
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        logger.info(f"  ✓ Batch {batch_num}/{total_batches} upserted")
    
    logger.info(f"\n✓ Successfully stored {len(vectors)} embeddings in Pinecone")
    logger.info(f"  Index: {index_name}")
    logger.info("="*70)
    
    return True


def main():
    logger.info("="*70)
    logger.info("LAB 1 STEP 3: EMBEDDING GENERATION")
    logger.info("="*70)
    
    # Allow strategy override via environment variable
    strategy = os.getenv("CHUNK_STRATEGY", DEFAULT_STRATEGY)
    logger.info(f"\nUsing chunking strategy: {strategy}")
    
    # Load chunks
    try:
        chunks = load_chunks(strategy)
    except FileNotFoundError as e:
        logger.error(f"\n❌ {e}")
        logger.error("\nAvailable strategies:")
        for f in sorted(CHUNKS_DIR.glob("*_chunks.json")):
            strategy_name = f.stem.replace('_chunks', '')
            logger.error(f"  - {strategy_name}")
        logger.error(f"\nTo use a different strategy:")
        logger.error(f'  $env:CHUNK_STRATEGY="strategy_name"')
        return
    
    # Generate embeddings
    try:
        embedded_chunks, metadata = embed_chunks(chunks, strategy)
    except ValueError as e:
        logger.error(f"\n❌ {e}")
        return
    except Exception as e:
        logger.error(f"\n❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Store in vector databases
    chromadb_success = store_in_chromadb(embedded_chunks, collection_name="fintbx")
    pinecone_success = store_in_pinecone(embedded_chunks, index_name="fintbx")
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("✓ EMBEDDING GENERATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"📊 Strategy: {strategy}")
    logger.info(f"📦 Chunks embedded: {metadata['num_chunks']}")
    logger.info(f"🔢 Model: {metadata['model']}")
    logger.info(f"📏 Dimensions: {metadata['dimensions']}")
    logger.info(f"⏱️  Generation time: {metadata['generation_time_sec']:.2f}s")
    logger.info(f"💰 Estimated cost: ${metadata['estimated_cost_usd']:.4f}")
    logger.info(f"\n📁 Output:")
    logger.info(f"   - Embeddings JSON: {EMBEDDINGS_DIR / f'{strategy}_embeddings.json'}")
    logger.info(f"   - Metadata: {EMBEDDINGS_DIR / f'{strategy}_embedding_metadata.json'}")
    
    if chromadb_success:
        logger.info(f"   - ChromaDB: {PROJECT_ROOT / 'data' / 'chromadb'}")
    if pinecone_success:
        logger.info(f"   - Pinecone: Index 'fintbx'")
    
    logger.info("")
    logger.info("💡 NEXT STEPS:")
    if chromadb_success or pinecone_success:
        logger.info("  ✅ Lab 1 is COMPLETE!")
        logger.info("  📝 Document your results in the Codelab")
        logger.info("  🚀 Move to Lab 2: Airflow orchestration")
    else:
        logger.info("  ⚠️  Install vector database:")
        logger.info("     pip install chromadb")
        logger.info("     (or pip install pinecone-client)")
        logger.info("  🔄 Then re-run this script")
    logger.info("="*70)


if __name__ == "__main__":
    main()
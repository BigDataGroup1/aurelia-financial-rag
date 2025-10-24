"""
Lab 1 Step 3: Generate embeddings using text-embedding-3-large
Store in vector database (ChromaDB or Pinecone)

HYBRID A+B:
- Stream per-batch (lower memory, smaller /tmp usage)
- Upload to GCS periodically + in finally (survives retries/evictions)
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import time
import logging

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Paths (unchanged defaults; can be overridden by environment if needed)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", PROJECT_ROOT / "data" / "chunks"))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", PROJECT_ROOT / "data" / "embeddings"))
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))  # default for model

# Strategy (unchanged behavior)
DEFAULT_STRATEGY = os.getenv("CHUNK_STRATEGY", "6_code_aware_1200_200")  # same default

# ----------------------------------------------------------------------
# Streaming & upload knobs (env-tunable)
# ----------------------------------------------------------------------
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))            # ↓ from 100 → 50 lowers memory spike
SLEEP_BETWEEN = float(os.getenv("EMBED_SLEEP_SEC", "0.4"))       # small pause for rate smoothing
UPLOAD_EVERY_N = int(os.getenv("EMBED_UPLOAD_EVERY", "5"))       # upload to GCS every N batches
DELETE_LOCAL_AFTER_UPLOAD = os.getenv("EMBED_DELETE_LOCAL", "1") == "1"  # free ephemeral storage

# GCS (optional; if not set, local only)
GCS_BUCKET = os.getenv("GCS_BUCKET")  # e.g. "aurelia-rag-data"
# Use Airflow logical date if provided; otherwise fall back to "manual"
GCS_DS = os.getenv("AIRFLOW_CTX_EXECUTION_DATE") or os.getenv("AIRFLOW_CTX_DAG_RUN_EXECUTION_DATE") or os.getenv("DS") or "manual"
GCS_PREFIX = os.getenv("EMBED_GCS_PREFIX", f"embeddings/{GCS_DS}/")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _gcs_client():
    """Lazy import to keep dependency optional."""
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception as e:
        logger.warning(f"⚠️ google-cloud-storage not available: {e}. Skipping uploads.")
        return None

def _upload_dir_to_gcs(local_dir: Path, bucket_name: str, prefix: str,
                       patterns: Tuple[str, ...] = ("*.json", "*.ndjson", "*.parquet")) -> int:
    """
    Upload files matching patterns from local_dir to gs://bucket/prefix/.
    Returns number of files uploaded.
    """
    client = _gcs_client()
    if not client or not bucket_name:
        return 0

    bucket = client.bucket(bucket_name)
    count = 0
    for pat in patterns:
        for f in sorted(local_dir.glob(pat)):
            blob = bucket.blob(f"{prefix}{f.name}")
            try:
                blob.upload_from_filename(str(f))
                count += 1
            except Exception as e:
                logger.warning(f"Upload failed for {f.name}: {e}")
    if count:
        logger.info(f"✓ Uploaded {count} file(s) -> gs://{bucket_name}/{prefix}")
    return count

def load_chunks(strategy_name: str) -> List[Dict]:
    """Load chunks from JSON (unchanged)."""
    chunks_file = CHUNKS_DIR / f"{strategy_name}_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    logger.info(f"✓ Loaded {len(chunks)} chunks from {strategy_name}")
    return chunks

# ----------------------------------------------------------------------
# Core: streamed, per-batch generation + safe uploads
# ----------------------------------------------------------------------
def embed_chunks_streaming(chunks: List[Dict], strategy_name: str) -> Tuple[dict, list]:
    """
    Stream embeddings in batches, write combined JSON incrementally, and upload periodically.
    Returns (metadata_dict, batch_manifest_list)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set!"
        )
    client = OpenAI(api_key=api_key)

    total = len(chunks)
    logger.info("\n" + "="*70)
    logger.info("GENERATING EMBEDDINGS (streamed)")
    logger.info("-"*70)
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Dimensions: {EMBEDDING_DIMENSIONS}")
    logger.info(f"Total chunks: {total}")

    # Cost estimate (unchanged)
    total_chars = sum(c['char_count'] for c in chunks)
    est_tokens = total_chars / 4
    est_cost = (est_tokens / 1_000_000) * 0.13
    logger.info(f"Estimated tokens: ~{est_tokens:,.0f}")
    logger.info(f"Estimated cost: ~${est_cost:.4f}")
    logger.info("-"*70)

    # Paths
    combined_tmp = EMBEDDINGS_DIR / f"{strategy_name}_embeddings.json.tmp"
    combined_final = EMBEDDINGS_DIR / f"{strategy_name}_embeddings.json"
    metadata_file = EMBEDDINGS_DIR / f"{strategy_name}_embedding_metadata.json"

    # Ensure clean temp file
    if combined_tmp.exists():
        combined_tmp.unlink(missing_ok=True)

    start_time = time.time()
    batch_manifest = []  # record per-batch ndjson files

    # ---------------------------
    # try/finally: ALWAYS upload whatever exists if we die mid-run
    # ---------------------------
    uploaded_once = False
    try:
        # Open combined file for streaming JSON array
        with open(combined_tmp, "w", encoding="utf-8") as out:
            out.write("[\n")
            first_written = True

            for i in range(0, total, BATCH_SIZE):
                batch_idx = i // BATCH_SIZE + 1
                end = min(i + BATCH_SIZE, total)
                texts = [c["content"] for c in chunks[i:end]]

                logger.info(f"  Processing batch {batch_idx}/{(total - 1)//BATCH_SIZE + 1} ({len(texts)} chunks)...")
                resp = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
                vectors = [item.embedding for item in resp.data]

                # Build per-batch items & write:
                batch_items = []
                for j, emb in enumerate(vectors):
                    c = chunks[i + j]
                    item = {
                        "chunk_id": c["chunk_id"],
                        "content": c["content"],
                        "metadata": c["metadata"],
                        "char_count": c["char_count"],
                        "embedding": emb,
                        "embedding_model": EMBEDDING_MODEL,
                        "embedding_dimensions": len(emb),
                    }
                    batch_items.append(item)

                # 1) Append to combined JSON (without holding all in RAM)
                for k, item in enumerate(batch_items):
                    if not first_written:
                        out.write(",\n")
                    out.write(json.dumps(item, ensure_ascii=False))
                    first_written = False
                out.flush()

                # 2) Also write an ndjson file per batch (great for partial uploads)
                nd = EMBEDDINGS_DIR / f"{strategy_name}_embeddings_batch_{batch_idx:04d}.ndjson"
                with open(nd, "w", encoding="utf-8") as ndout:
                    for item in batch_items:
                        ndout.write(json.dumps(item, ensure_ascii=False) + "\n")
                batch_manifest.append(nd.name)

                logger.info(f"  ✓ Batch {batch_idx} complete → wrote {nd.name}")

                # Upload every N batches to GCS (if configured)
                if GCS_BUCKET and (batch_idx % UPLOAD_EVERY_N == 0):
                    _upload_dir_to_gcs(EMBEDDINGS_DIR, GCS_BUCKET, GCS_PREFIX, patterns=("*.ndjson",))
                    uploaded_once = True
                    if DELETE_LOCAL_AFTER_UPLOAD:
                        try:
                            # Delete only per-batch files (keep combined tmp)
                            for ndjson_file in EMBEDDINGS_DIR.glob(f"{strategy_name}_embeddings_batch_*.ndjson"):
                                ndjson_file.unlink(missing_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed deleting local ndjson batch files: {e}")

                if i + BATCH_SIZE < total and SLEEP_BETWEEN > 0:
                    time.sleep(SLEEP_BETWEEN)

            # Close the JSON array
            out.write("\n]\n")

        # Atomically move final combined file
        os.replace(combined_tmp, combined_final)
        logger.info(f"✓ Combined JSON written: {combined_final}")

        # Upload combined + any remaining ndjson files at end
        if GCS_BUCKET:
            _upload_dir_to_gcs(EMBEDDINGS_DIR, GCS_BUCKET, GCS_PREFIX, patterns=("*.json", "*.ndjson"))
            uploaded_once = True
            if DELETE_LOCAL_AFTER_UPLOAD:
                try:
                    for ndjson_file in EMBEDDINGS_DIR.glob(f"{strategy_name}_embeddings_batch_*.ndjson"):
                        ndjson_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed deleting local ndjson batch files: {e}")

    finally:
        # SAFETY NET: if we never uploaded (e.g., crash before end), push what we have
        if GCS_BUCKET and not uploaded_once:
            logger.info("↪ Task ending early — uploading partial files now (finally).")
            _upload_dir_to_gcs(EMBEDDINGS_DIR, GCS_BUCKET, GCS_PREFIX, patterns=("*.json", "*.ndjson"))

    elapsed = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info(f"✓ Generated embeddings in {elapsed:.2f}s")
    logger.info(f"  Avg time per chunk: {elapsed / max(1, total):.3f}s")
    logger.info(f"  Throughput: {total / max(0.001, elapsed):.1f} chunks/sec")
    logger.info("="*70)

    # Metadata (unchanged keys so downstream stays happy)
    meta = {
        "strategy": strategy_name,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "num_chunks": total,
        "total_chars": total_chars,
        "generation_time_sec": elapsed,
        "avg_time_per_chunk": (elapsed / max(1, total)),
        "estimated_tokens": int(est_tokens),
        "estimated_cost_usd": round(est_cost, 4),
        "gcs_bucket": GCS_BUCKET or "",
        "gcs_prefix": GCS_PREFIX if GCS_BUCKET else "",
        "batch_size": BATCH_SIZE,
        "upload_every_n_batches": UPLOAD_EVERY_N,
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_file.name}")

    # Final upload for metadata (idempotent)
    if GCS_BUCKET:
        _upload_dir_to_gcs(EMBEDDINGS_DIR, GCS_BUCKET, GCS_PREFIX, patterns=("*.json",))
    return meta, batch_manifest

# ----------------------------------------------------------------------
# Optional DB sinks (unchanged)
# ----------------------------------------------------------------------
def store_in_chromadb(embedded_json_path: Path, collection_name: str = "fintbx") -> bool:
    """
    Load combined JSON and store it to ChromaDB. (Same behavior; reads from file)
    """
    try:
        import chromadb
    except ImportError:
        logger.warning("\n⚠️  ChromaDB not installed. Run: pip install chromadb")
        logger.warning("   Skipping ChromaDB storage...")
        return False

    if not embedded_json_path.exists():
        logger.warning(f"Combined embeddings not found: {embedded_json_path}")
        return False

    logger.info("\n" + "="*70)
    logger.info("STORING IN CHROMADB")
    logger.info("-"*70)

    data = json.loads(embedded_json_path.read_text(encoding="utf-8"))
    if not data:
        logger.warning("No embeddings to store.")
        return False

    chroma_path = PROJECT_ROOT / "data" / "chromadb"
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Replace collection each run (as before)
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Financial Toolbox embeddings"}
    )
    logger.info(f"✓ Created collection: {collection_name}")

    ids = [f"chunk_{c['chunk_id']}" for c in data]
    embeddings = [c['embedding'] for c in data]
    documents = [c['content'] for c in data]
    metadatas = [c['metadata'] for c in data]

    bs = 100
    total = len(data)
    logger.info(f"Storing {total} vectors in {(total-1)//bs+1} batch(es)...")
    for i in range(0, total, bs):
        collection.add(
            ids=ids[i:i+bs],
            embeddings=embeddings[i:i+bs],
            documents=documents[i:i+bs],
            metadatas=metadatas[i:i+bs]
        )
        logger.info(f"  ✓ Batch {(i//bs)+1}/{(total-1)//bs+1} stored")

    logger.info(f"\n✓ Successfully stored {total} embeddings in ChromaDB")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Path: {chroma_path}")
    logger.info("="*70)
    return True


def store_in_pinecone(embedded_json_path: Path, index_name: str = "fintbx") -> bool:
    """
    Load combined JSON and store it to Pinecone (same behavior; reads from file).
    """
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        logger.warning("\n⚠️  Pinecone not installed. Run: pip install pinecone-client")
        logger.warning("   Skipping Pinecone storage...")
        return False

    if not embedded_json_path.exists():
        logger.warning(f"Combined embeddings not found: {embedded_json_path}")
        return False

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning("PINECONE_API_KEY not set. Skipping Pinecone storage.")
        return False

    logger.info("\n" + "="*70)
    logger.info("STORING IN PINECONE")
    logger.info("-"*70)

    data = json.loads(embedded_json_path.read_text(encoding="utf-8"))
    if not data:
        logger.warning("No embeddings to upsert.")
        return False

    pc = Pinecone(api_key=api_key)
    existing = pc.list_indexes().names()
    if index_name not in existing:
        logger.info(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"✓ Created index: {index_name}")
        time.sleep(10)
    else:
        logger.info(f"Using existing index: {index_name}")

    index = pc.Index(index_name)

    # Truncate content in metadata per Pinecone limits
    vectors = []
    for c in data:
        md = {**c["metadata"]}
        md.update({
            "content_preview": c["content"][:1000],
            "char_count": c["char_count"],
            "chunk_id": c["chunk_id"],
        })
        vectors.append({"id": f"chunk_{c['chunk_id']}", "values": c["embedding"], "metadata": md})

    bs = 100
    total = len(vectors)
    logger.info(f"Upserting {total} vectors in {(total-1)//bs+1} batch(es)...")
    for i in range(0, total, bs):
        index.upsert(vectors=vectors[i:i+bs])
        logger.info(f"  ✓ Batch {(i//bs)+1}/{(total-1)//bs+1} upserted")

    logger.info(f"\n✓ Successfully stored {total} embeddings in Pinecone")
    logger.info(f"  Index: {index_name}")
    logger.info("="*70)
    return True

# ----------------------------------------------------------------------
# Main (unchanged external behavior)
# ----------------------------------------------------------------------
def main():
    logger.info("="*70)
    logger.info("LAB 1 STEP 3: EMBEDDING GENERATION")
    logger.info("="*70)

    strategy = os.getenv("CHUNK_STRATEGY", DEFAULT_STRATEGY)
    logger.info(f"\nUsing chunking strategy: {strategy}")

    # Load chunks (unchanged)
    try:
        chunks = load_chunks(strategy)
    except FileNotFoundError as e:
        logger.error(f"\n❌ {e}")
        logger.error("\nAvailable strategies:")
        for f in sorted(CHUNKS_DIR.glob("*_chunks.json")):
            strategy_name = f.stem.replace('_chunks', '')
            logger.error(f"  - {strategy_name}")
        logger.error('\nTo use a different strategy:')
        logger.error('  $env:CHUNK_STRATEGY="strategy_name"')
        return

    # Generate embeddings (streaming + resilient upload)
    try:
        meta, manifest = embed_chunks_streaming(chunks, strategy)
    except Exception as e:
        logger.error(f"\n❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Store in vector DBs (optional; unchanged APIs)
    combined_file = EMBEDDINGS_DIR / f"{strategy}_embeddings.json"
    chromadb_success = store_in_chromadb(combined_file, collection_name="fintbx")
    pinecone_success = store_in_pinecone(combined_file, index_name="fintbx")

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✓ EMBEDDING GENERATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"📊 Strategy: {strategy}")
    logger.info(f"📦 Chunks embedded: {meta['num_chunks']}")
    logger.info(f"🔢 Model: {meta['model']}")
    logger.info(f"📏 Dimensions: {meta['dimensions']}")
    logger.info(f"⏱️  Generation time: {meta['generation_time_sec']:.2f}s")
    logger.info(f"💰 Estimated cost: ${meta['estimated_cost_usd']:.4f}")
    logger.info("\n📁 Output:")
    logger.info(f"   - Embeddings JSON: {combined_file}")
    logger.info(f"   - Metadata: {EMBEDDINGS_DIR / f'{strategy}_embedding_metadata.json'}")
    if GCS_BUCKET:
        logger.info(f"   - GCS: gs://{GCS_BUCKET}/{GCS_PREFIX} (combined + per-batch)")
    if chromadb_success:
        logger.info(f"   - ChromaDB: {PROJECT_ROOT / 'data' / 'chromadb'}")
    if pinecone_success:
        logger.info("   - Pinecone: Index 'fintbx'")
    logger.info("="*70)


if __name__ == "__main__":
    main()

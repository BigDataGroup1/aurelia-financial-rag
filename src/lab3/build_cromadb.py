"""
Build ChromaDB locally from GCS embeddings
IMPORTANT: Must use same ChromaDB version as App Engine (0.4.22)
"""
import json
import chromadb
from google.cloud import storage
from pathlib import Path
import shutil

# ✅ VERIFY VERSION
import pkg_resources
chromadb_version = pkg_resources.get_distribution("chromadb").version
print(f"ChromaDB Version: {chromadb_version}")

if not chromadb_version.startswith("0.4."):
    print("❌ ERROR: Wrong ChromaDB version!")
    print(f"   Current: {chromadb_version}")
    print(f"   Required: 0.4.22")
    print("\nRun: pip install chromadb==0.4.22")
    exit(1)

print("="*70)
print("BUILDING CHROMADB LOCALLY")
print("="*70)

# Download embeddings from GCS
print("\n1. Downloading embeddings from GCS...")
client = storage.Client()
bucket = client.bucket('aurelia-rag-data')

# Use latest date
blob = bucket.blob('embeddings/2025-10-24/6_code_aware_1200_200_embeddings.json')
blob.reload()

if blob.size:
    print(f"   File size: {blob.size / 1024 / 1024:.1f} MB")
else:
    print("   File size: Unknown")
print("   This may take 2-5 minutes...")

embeddings_json = blob.download_as_text()
embedded_chunks = json.loads(embeddings_json)

print(f"✅ Downloaded {len(embedded_chunks)} chunks")

# Build ChromaDB
print("\n2. Building ChromaDB...")
output_path = Path("../../data/chromadb_upload")
output_path.mkdir(parents=True, exist_ok=True)

# Delete existing if present
if (output_path / "chroma.sqlite3").exists():
    shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=str(output_path))

# Create collection (NO embedding_function for pre-computed!)
collection = chroma_client.create_collection(
    name="fintbx",
    metadata={
        "embedding_model": "text-embedding-3-large",
        "chromadb_version": chromadb_version,
        "created_from": "6_code_aware_1200_200_embeddings.json",
        "date": "2025-10-24"
    },
    embedding_function=None  # ✅ CRITICAL!
)

print(f"   Adding {len(embedded_chunks)} vectors...")

# Add in batches
batch_size = 100
for i in range(0, len(embedded_chunks), batch_size):
    batch = embedded_chunks[i:i+batch_size]
    
    collection.add(
        ids=[f"chunk_{c['chunk_id']}" for c in batch],
        embeddings=[c['embedding'] for c in batch],
        documents=[c['content'] for c in batch],
        metadatas=[c['metadata'] for c in batch]
    )
    
    if (i // batch_size + 1) % 5 == 0:
        print(f"   Batch {i // batch_size + 1}/{(len(embedded_chunks)-1)//batch_size+1}")

print(f"✅ ChromaDB built with {collection.count()} vectors")

# Upload to GCS
print("\n3. Uploading ChromaDB to GCS...")
print(f"   Uploading from: {output_path}")

# Upload all files
for file_path in output_path.rglob('*'):
    if file_path.is_file():
        relative_path = file_path.relative_to(output_path)
        gcs_path = f'chromadb/2025-10-24/{relative_path}'
        
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(file_path))
        
        print(f"   ✅ Uploaded: {gcs_path}")

print(f"\n✅ ChromaDB uploaded to: gs://aurelia-rag-data/chromadb/2025-10-24/")
print("="*70)
print("\nNext step:")
print("  Deploy: gcloud app deploy")
print("="*70)
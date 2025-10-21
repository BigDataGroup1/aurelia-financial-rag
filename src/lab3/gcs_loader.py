"""
GCS Data Loader - Downloads embeddings from GCS and rebuilds ChromaDB locally
Required for App Engine deployment since local chromadb/ folder doesn't exist in cloud
"""
import logging
import json
from pathlib import Path
from google.cloud import storage
import chromadb

logger = logging.getLogger(__name__)


def load_embeddings_from_gcs(bucket_name: str, date_folder: str = None):
    """
    Download embeddings JSON from GCS
    
    Args:
        bucket_name: GCS bucket name (e.g., 'aurelia-rag-data')
        date_folder: Specific date folder or None for latest
        
    Returns:
        List of embedded chunks
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Find latest embeddings if date not specified
        if not date_folder:
            logger.info("Finding latest embeddings folder...")
            blobs = list(bucket.list_blobs(prefix="embeddings/"))
            date_folders = set()
            
            for blob in blobs:
                parts = blob.name.split('/')
                if len(parts) >= 2 and parts[0] == "embeddings":
                    date_folders.add(parts[1])
            
            if not date_folders:
                raise Exception("No embeddings found in GCS")
            
            date_folder = max(date_folders)
            logger.info(f"Using latest: {date_folder}")
        
        # Download embeddings JSON
        embeddings_path = f"embeddings/{date_folder}/6_code_aware_1200_200_embeddings.json"
        blob = bucket.blob(embeddings_path)
        
        if not blob.exists():
            raise Exception(f"Embeddings not found: {embeddings_path}")
        
        logger.info(f"Downloading: {embeddings_path}")
        logger.info("This may take 1-2 minutes (70MB file)...")
        
        embeddings_json = json.loads(blob.download_as_text())
        
        logger.info(f"✓ Downloaded {len(embeddings_json)} embedded chunks")
        return embeddings_json
        
    except Exception as e:
        logger.error(f"Failed to load embeddings from GCS: {e}")
        raise


def initialize_chromadb(embeddings_data: list, chromadb_path: Path, collection_name: str = "fintbx"):
    """
    Initialize ChromaDB with embeddings from GCS
    
    Args:
        embeddings_data: List of embedded chunks from GCS
        chromadb_path: Local path for ChromaDB (/tmp/chromadb)
        collection_name: Collection name
        
    Returns:
        ChromaDB client
    """
    try:
        # Create directory
        chromadb_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {chromadb_path}")
        
        # Create client
        chroma_client = chromadb.PersistentClient(path=str(chromadb_path))
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"description": "Financial Toolbox embeddings from GCS"}
        )
        
        logger.info(f"Loading {len(embeddings_data)} vectors into ChromaDB...")
        
        # Prepare data
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in embeddings_data]
        embeddings = [chunk['embedding'] for chunk in embeddings_data]
        documents = [chunk['content'] for chunk in embeddings_data]
        metadatas = [chunk['metadata'] for chunk in embeddings_data]
        
        # Add in batches
        batch_size = 100
        total_batches = (len(embeddings_data) - 1) // batch_size + 1
        
        for i in range(0, len(embeddings_data), batch_size):
            end_idx = min(i + batch_size, len(embeddings_data))
            
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            batch_num = i // batch_size + 1
            logger.info(f"  Loaded batch {batch_num}/{total_batches}")
        
        count = collection.count()
        logger.info(f"✓ ChromaDB initialized with {count} vectors")
        
        return chroma_client
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise


def ensure_chromadb_ready(bucket_name: str, chromadb_path: Path):
    """
    Ensure ChromaDB is loaded and ready
    Called during FastAPI startup
    
    This function:
    1. Checks if ChromaDB already exists locally
    2. If not, downloads embeddings from GCS
    3. Initializes ChromaDB with the embeddings
    
    Args:
        bucket_name: GCS bucket name
        chromadb_path: Local path for ChromaDB
    """
    logger.info("="*70)
    logger.info("CHROMADB INITIALIZATION")
    logger.info("="*70)
    
    # Check if already loaded
    if chromadb_path.exists():
        try:
            client = chromadb.PersistentClient(path=str(chromadb_path))
            collection = client.get_collection(name="fintbx")
            count = collection.count()
            
            if count > 0:
                logger.info(f"✓ ChromaDB already loaded ({count} vectors)")
                logger.info("="*70)
                return client
        except:
            logger.info("ChromaDB exists but collection not found, reloading...")
    
    # Not loaded - download and initialize
    logger.info("ChromaDB not found locally")
    logger.info("Loading from GCS (this happens once on first startup)...")
    logger.info("="*70)
    
    # Download embeddings
    embeddings_data = load_embeddings_from_gcs(bucket_name)
    
    # Initialize ChromaDB
    client = initialize_chromadb(embeddings_data, chromadb_path)
    
    logger.info("="*70)
    logger.info("✓ CHROMADB READY")
    logger.info("="*70)
    
    return client
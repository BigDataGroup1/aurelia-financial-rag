"""
AURELIA Financial RAG Service - Main FastAPI Application
Lab 3: FastAPI RAG Service with Retrieval, Generation, and Caching
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

from config import settings
from models import (
    QueryRequest, QueryResponse,
    SeedRequest, SeedResponse,
    HealthResponse,
    GenerationContext
)
from services import (
    get_embedding_service,
    get_vector_store_service,
    get_wikipedia_service,
    get_generation_service
)
from database.cache import get_cache_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing AURELIA RAG Service...")

"""
Add these functions to src/lab3/main.py
Place them BEFORE the endpoint definitions, around line 60
"""

def get_latest_date_from_gcs(bucket_name: str, prefix: str) -> str:
    """
    Auto-detect latest date folder in GCS bucket
    
    Args:
        bucket_name: GCS bucket name
        prefix: Folder prefix (e.g., 'embeddings/')
        
    Returns:
        Latest date string (e.g., '2025-10-21')
    """
    try:
        from google.cloud import storage
        from datetime import datetime
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List all blobs with the prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        # Extract unique date folders
        dates = set()
        for blob in blobs:
            # blob.name format: 'embeddings/2025-10-21/embeddings.json'
            # Split and get the date part
            parts = blob.name.split('/')
            if len(parts) >= 2 and parts[0] == prefix.rstrip('/'):
                date_str = parts[1]
                # Validate it's a date format (YYYY-MM-DD)
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates.add(date_str)
                except ValueError:
                    continue
        
        if not dates:
            raise ValueError(f"No date folders found in gs://{bucket_name}/{prefix}")
        
        # Return latest date
        latest = sorted(dates)[-1]
        logger.info(f"Auto-detected latest date: {latest}")
        return latest
        
    except Exception as e:
        logger.error(f"Failed to detect latest date from GCS: {e}")
        raise

def download_and_build_chromadb_from_gcs(
    bucket_name: str,
    date: str = None,
    local_chromadb_path: str = "/tmp/chromadb"
) -> str:
    """
    Download embeddings from GCS and build ChromaDB locally
    
    Args:
        bucket_name: GCS bucket name
        date: Date folder (auto-detected if None)
        local_chromadb_path: Where to create ChromaDB
        
    Returns:
        Path to created ChromaDB directory
    """
    import json
    import os
    from pathlib import Path
    
    try:
        from google.cloud import storage
        import chromadb
        
        logger.info("="*70)
        logger.info("BUILDING CHROMADB FROM GCS EMBEDDINGS")
        logger.info("="*70)
        
        # Auto-detect latest date if not provided
        if date is None:
            date = get_latest_date_from_gcs(bucket_name, 'embeddings/')
        
        logger.info(f"Using date: {date}")
        
        # Download embeddings.json from GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        embeddings_path = f'embeddings/{date}/6_code_aware_1200_200_embeddings.json'
        blob = bucket.blob(embeddings_path)

        if not blob.exists():
            raise FileNotFoundError(f"Embeddings not found: gs://{bucket_name}/{embeddings_path}")

        logger.info(f"Downloading: gs://{bucket_name}/{embeddings_path}")

        # Reload blob to get size
        blob.reload()
        if blob.size:
            logger.info(f"Size: {blob.size / 1024 / 1024:.1f} MB")
        else:
            logger.info("Size: Unknown (metadata not available)")   
        
        # Download to temp file
        temp_embeddings = "/tmp/embeddings_download.json"
        blob.download_to_filename(temp_embeddings)
        logger.info(f"✓ Downloaded to {temp_embeddings}")
        
        # Load embeddings JSON
        logger.info("Loading embeddings JSON...")
        with open(temp_embeddings, 'r') as f:
            embedded_chunks = json.load(f)
        
        logger.info(f"✓ Loaded {len(embedded_chunks)} embedded chunks")
        
        # Create ChromaDB directory
        os.makedirs(local_chromadb_path, exist_ok=True)
        logger.info(f"Creating ChromaDB at: {local_chromadb_path}")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=local_chromadb_path)
        
        # Delete collection if exists
        try:
            chroma_client.delete_collection(name="fintbx")
            logger.info("Deleted existing collection")
        except:
            pass
        
        # Create collection (CRITICAL: embedding_function=None for pre-computed embeddings)
        collection = chroma_client.create_collection(
            name="fintbx",
            metadata={
                "embedding_model": "text-embedding-3-large",
                "embedding_dimensions": 3072,
                "created_from": f"gs://{bucket_name}/{embeddings_path}"
            },
            embedding_function=None  # Pre-computed embeddings!
        )
        logger.info("✓ Created collection: fintbx")
        
        # Add vectors in batches
        batch_size = 100
        total_batches = (len(embedded_chunks) - 1) // batch_size + 1
        
        logger.info(f"Adding {len(embedded_chunks)} vectors in {total_batches} batches...")
        
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i+batch_size]
            
            collection.add(
                ids=[f"chunk_{c['chunk_id']}" for c in batch],
                embeddings=[c['embedding'] for c in batch],
                documents=[c['content'] for c in batch],
                metadatas=[c['metadata'] for c in batch]
            )
            
            batch_num = i // batch_size + 1
            if batch_num % 5 == 0 or batch_num == total_batches:
                logger.info(f"  ✓ Batch {batch_num}/{total_batches}")
        
        # Verify
        count = collection.count()
        logger.info(f"\n✓ ChromaDB created successfully!")
        logger.info(f"  Vectors: {count}")
        logger.info(f"  Location: {local_chromadb_path}")
        logger.info("="*70)
        
        # Clean up temp file
        os.remove(temp_embeddings)
        
        return local_chromadb_path
        
    except Exception as e:
        logger.error(f"Failed to build ChromaDB from GCS: {e}")
        raise RuntimeError(f"ChromaDB build failed: {str(e)}")


def initialize_chromadb_for_cloud():
    """
    Initialize ChromaDB based on environment (local vs cloud)
    
    Returns:
        Path to ChromaDB directory
    """
    import os
    from pathlib import Path
    
    # Check if running in cloud (GCS bucket configured)
    gcs_bucket = os.getenv('CHROMADB_GCS_BUCKET')
    
    if gcs_bucket:
        # Running in cloud - build from GCS embeddings
        logger.info("Cloud environment detected - building ChromaDB from GCS")
        
        # Auto-detect latest date or use specified date
        date = os.getenv('EMBEDDINGS_DATE', None)  # Optional: specify date
        
        chromadb_path = download_and_build_chromadb_from_gcs(
            bucket_name=gcs_bucket,
            date=date,
            local_chromadb_path="/tmp/chromadb"
        )
        
        return chromadb_path
    
    else:
        # Running locally - use local ChromaDB
        logger.info("Local environment detected - using local ChromaDB")
        local_path = str(settings.chromadb_path)
        
        if not Path(local_path).exists():
            raise RuntimeError(f"Local ChromaDB not found: {local_path}")
        
        logger.info(f"Using local ChromaDB: {local_path}")
        return local_path
# ============================================================================
# Core RAG Logic
# ============================================================================
def load_concepts_from_json(file_path_or_gcs: str = None) -> int:
    """
    Load pre-generated concept definitions from JSON file into cache
    Service continues normally even if loading fails
    
    Args:
        file_path_or_gcs: Local file path or GCS URI (optional)
        
    Returns:
        Number of concepts loaded into cache (0 if failed)
    """
    import json
    from pathlib import Path
    
    try:
        cache_service = get_cache_service()
    except Exception as e:
        logger.warning(f"Cache service not available: {e}")
        return 0
    
    loaded_count = 0
    concepts_data = None
    
    try:
        # Determine where to load from
        if file_path_or_gcs is None:
            # Default: Try local first, then GCS
            local_path = settings.project_root / "data" / "concepts" / "concept_definitions.json"
            
            if local_path.exists():
                logger.info(f"Loading concepts from local file: {local_path}")
                try:
                    with open(local_path, 'r', encoding='utf-8') as f:
                        concepts_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {local_path}: {e}")
                    return 0
                except Exception as e:
                    logger.warning(f"Could not read {local_path}: {e}")
                    return 0
            else:
                logger.info(f"Local concept file not found: {local_path}")
                logger.info("Trying GCS (if configured)...")
                
                # Try GCS (gracefully fail if not configured)
                try:
                    from google.cloud import storage
                    from datetime import datetime
                    
                    client = storage.Client()
                    bucket = client.bucket('aurelia-rag-data')
                    
                    # Try today's date
                    today = datetime.now().strftime('%Y-%m-%d')
                    blob = bucket.blob(f'concepts/{today}/concept_definitions.json')
                    
                    if blob.exists():
                        concepts_json = blob.download_as_text()
                        concepts_data = json.loads(concepts_json)
                        logger.info(f"Loaded from GCS: concepts/{today}/concept_definitions.json")
                    else:
                        logger.info(f"GCS file not found: concepts/{today}/concept_definitions.json")
                        return 0
                        
                except ImportError:
                    logger.info("GCS library not installed (google-cloud-storage)")
                    return 0
                except Exception as e:
                    logger.info(f"GCS not available or configured: {e}")
                    return 0
        
        elif file_path_or_gcs.startswith('gs://'):
            # GCS path provided explicitly
            logger.info(f"Loading concepts from GCS: {file_path_or_gcs}")
            try:
                from google.cloud import storage
                
                # Parse GCS URI: gs://bucket/path/to/file.json
                parts = file_path_or_gcs.replace('gs://', '').split('/', 1)
                bucket_name = parts[0]
                blob_path = parts[1] if len(parts) > 1 else ''
                
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                if blob.exists():
                    concepts_json = blob.download_as_text()
                    concepts_data = json.loads(concepts_json)
                else:
                    logger.warning(f"GCS blob not found: {file_path_or_gcs}")
                    return 0
                    
            except Exception as e:
                logger.warning(f"Failed to load from GCS: {e}")
                return 0
        
        else:
            # Local file path provided explicitly
            logger.info(f"Loading concepts from local file: {file_path_or_gcs}")
            try:
                with open(file_path_or_gcs, 'r', encoding='utf-8') as f:
                    concepts_data = json.load(f)
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path_or_gcs}")
                return 0
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                return 0
            except Exception as e:
                logger.warning(f"Could not read file: {e}")
                return 0
        
        # Validate we have data
        if not concepts_data:
            logger.info("No concept data to load (empty file)")
            return 0
        
        if not isinstance(concepts_data, list):
            logger.warning("Concept data must be a list of concept definitions")
            return 0
        
        # Load concepts into cache
        logger.info(f"Found {len(concepts_data)} concept(s) to load...")
        
        for idx, concept_dict in enumerate(concepts_data):
            try:
                # Validate required fields
                if 'concept_name' not in concept_dict:
                    logger.warning(f"Concept {idx+1} missing 'concept_name', skipping")
                    continue
                
                # Convert dict to ConceptNote (with validation)
                concept_note = ConceptNote(**concept_dict)
                
                # Save to cache
                success = cache_service.save(concept_note)
                if success:
                    loaded_count += 1
                    logger.debug(f"  ✓ Loaded: {concept_note.concept_name}")
                else:
                    logger.warning(f"  ✗ Failed to cache: {concept_dict.get('concept_name')}")
                    
            except Exception as e:
                concept_name = concept_dict.get('concept_name', f'concept_{idx+1}')
                logger.warning(f"Failed to load '{concept_name}': {e}")
        
        logger.info(f"✓ Successfully loaded {loaded_count}/{len(concepts_data)} concepts into cache")
        return loaded_count
        
    except Exception as e:
        logger.error(f"Unexpected error loading concepts: {e}")
        return 0
def is_finance_related(concept: str) -> bool:
    """
    Check if a concept is related to finance/economics
    Uses OpenAI to classify the query
    
    Returns:
        True if finance-related, False otherwise
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a classifier that determines if a query is related to finance, economics, or financial markets.

Finance-related topics include: financial instruments, investment strategies, risk management, portfolio theory, derivatives, bonds, stocks, options, economic indicators, financial modeling, quantitative finance, corporate finance, banking, etc.

Respond with ONLY 'yes' or 'no'."""
                },
                {
                    "role": "user",
                    "content": f"Is this concept related to finance or economics? Concept: {concept}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip().lower()
        is_relevant = answer == 'yes'
        
        logger.info(f"Finance relevance check for '{concept}': {is_relevant}")
        return is_relevant
        
    except Exception as e:
        logger.warning(f"Finance relevance check failed: {e}. Assuming relevant.")
        return True  # Default to True if check fails
def retrieve_context(concept: str):
    """
    Retrieve context for a concept from ChromaDB or Wikipedia fallback
    Only for finance-related queries
    """
    # Get services
    embedding_service = get_embedding_service()
    vector_store = get_vector_store_service()
    wikipedia_service = get_wikipedia_service()
    
    # Step 1: Embed query
    query_embedding = embedding_service.embed_query(concept)
    
    # Step 2: Query ChromaDB FIRST
    chromadb_results = vector_store.query(query_embedding)
    
    # Step 3: Check if results are good enough
    if chromadb_results.documents and chromadb_results.similarities:
        max_similarity = max(chromadb_results.similarities)
        logger.info(f"ChromaDB max similarity: {max_similarity:.3f}")
        
        if max_similarity >= settings.similarity_threshold:
            logger.info(f"Using ChromaDB results (above threshold {settings.similarity_threshold})")
            return chromadb_results
    
    # Step 4: No good match in PDF - check if query is finance-related
    logger.info(f"No good PDF match for '{concept}'. Checking finance relevance...")
    
    if not is_finance_related(concept):
        logger.warning(f"Rejecting non-finance query: '{concept}'")
        raise HTTPException(
            status_code=400,
            detail=f"This service only handles finance-related concepts. '{concept}' is not found in fintbx.pdf and is not relevant to finance or economics."
        )
    
    # Step 5: Finance-related but not in PDF - use Wikipedia fallback
    logger.warning(
        f"ChromaDB results below threshold. "
        f"Falling back to Wikipedia for finance concept '{concept}'"
    )
    try:
        wikipedia_results = wikipedia_service.search(concept)
        return wikipedia_results
    except Exception as e:
        logger.error(f"Wikipedia fallback also failed: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"No information found for financial concept '{concept}' in either fintbx.pdf or Wikipedia"
        )

def generate_note(concept: str, force_refresh: bool = False):
    """
    Generate a concept note with caching support
    
    Args:
        concept: Concept name to generate note for
        force_refresh: If True, bypass cache
        
    Returns:
        ConceptNote and metadata (note, chunks, time_ms, cached)
    """
    start_time = time.time()
    cache_service = get_cache_service()
    
    # Check cache first (unless force_refresh)
    if not force_refresh:
        cached_note = cache_service.get(concept)
        if cached_note:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Returning cached note for '{concept}' ({elapsed_ms:.1f}ms)")
            return cached_note, 0, elapsed_ms, True
    
    # Cache miss or force refresh - generate new note
    # Retrieve context
    retrieval_result = retrieve_context(concept)
    
    # Prepare generation context
    generation_context = GenerationContext(
        concept=concept,
        context=retrieval_result.get_context(max_length=settings.max_context_tokens),
        source=retrieval_result.source,
        page_references=retrieval_result.get_page_references(),
        confidence=retrieval_result.calculate_confidence()
    )
    
    # Generate note
    generation_service = get_generation_service()
    concept_note = generation_service.generate_concept_note(generation_context)
    
    # Save to cache
    cache_service.save(concept_note)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return concept_note, len(retrieval_result.documents), elapsed_ms, False

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "query": "POST /query",
            "seed": "POST /seed",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Returns status of all service components
    """
    try:
        # Check vector store
        vector_store = get_vector_store_service()
        chromadb_healthy = vector_store.health_check()
        vector_count = vector_store.collection_size if chromadb_healthy else None
        
        # Check OpenAI
        generation_service = get_generation_service()
        openai_healthy = generation_service.health_check()
        
        # Check cache
        cache_service = get_cache_service()
        cache_healthy = cache_service.health_check()
        cached_count = cache_service.count() if cache_healthy else None
        
        # Overall status
        all_healthy = chromadb_healthy and openai_healthy and cache_healthy
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            chromadb_status="connected" if chromadb_healthy else "disconnected",
            database_status="connected" if cache_healthy else "disconnected",  # CHANGED
            openai_status="connected" if openai_healthy else "disconnected",
            vector_count=vector_count,
            cached_concepts=cached_count  # CHANGED
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_concept(request: QueryRequest):
    """
    Query endpoint for concept note generation
    
    Process:
    1. Check cache (if not force_refresh)
    2. Embed query
    3. Query ChromaDB
    4. Fallback to Wikipedia if needed
    5. Generate structured note with LLM
    6. Cache result
    7. Return response
    """
    try:
        logger.info(f"Query received: concept='{request.concept}', force_refresh={request.force_refresh}")
        
        # Generate note
        concept_note, retrieved_chunks, generation_time_ms, cached = generate_note(
            concept=request.concept,
            force_refresh=request.force_refresh
        )
        
        return QueryResponse(
            concept_note=concept_note,
            retrieved_chunks=retrieved_chunks,
            generation_time_ms=generation_time_ms,
            cached=cached
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed for '{request.concept}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/seed", response_model=SeedResponse, tags=["Seed"])
async def seed_concepts(request: SeedRequest):
    """
    Seed endpoint for batch concept note generation
    
    Pre-generates and caches concept notes for a list of concepts.
    Used by Airflow DAG (Lab 2) or for manual batch processing.
    """
    try:
        logger.info(f"Seed request received: {len(request.concepts)} concepts")
        
        start_time = time.time()
        seeded = 0
        failed = 0
        failed_concepts = []
        
        for concept in request.concepts:
            try:
                logger.info(f"Seeding concept: {concept}")
                generate_note(concept, force_refresh=True)
                seeded += 1
            except Exception as e:
                logger.error(f"Failed to seed '{concept}': {e}")
                failed += 1
                failed_concepts.append(concept)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SeedResponse(
            total_concepts=len(request.concepts),
            seeded=seeded,
            failed=failed,
            failed_concepts=failed_concepts,
            generation_time_ms=elapsed_ms
        )
        
    except Exception as e:
        logger.error(f"Seed operation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Seed operation failed: {str(e)}"
        )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("AURELIA Financial RAG Service Starting...")
    
    # 🔍 STEP 1: Detect environment (local or cloud)
    import os
    gcs_bucket = os.getenv('CHROMADB_GCS_BUCKET')
    is_cloud = gcs_bucket is not None
    
    logger.info(f"Environment: {'CLOUD' if is_cloud else 'LOCAL'}")
    
    # 🏗️ STEP 2: Initialize ChromaDB based on environment
    chromadb_path = None
    
    if is_cloud:
        # CLOUD MODE: Build ChromaDB from GCS embeddings
        chromadb_path = download_and_build_chromadb_from_gcs(
            bucket_name=gcs_bucket,
            date=None  # Auto-detect latest date
        )
    else:
        # LOCAL MODE: Use existing ChromaDB
        chromadb_path = None  # Uses default from settings
    
    # 🚀 STEP 3: Initialize services (with ChromaDB path)
    get_embedding_service()
    get_vector_store_service(chromadb_path)  # ← Pass cloud path!
    get_wikipedia_service()
    get_generation_service()
    get_cache_service()
    
    # Initialize database
    from database.models import init_database
    init_database()
    
    # 📥 STEP 4: Load pre-seeded concepts (from GCS or local)
    if is_cloud:
        # Download from GCS
        date = get_latest_date_from_gcs(gcs_bucket, 'concepts/')
        concepts_path = f"gs://{gcs_bucket}/concepts/{date}/concept_definitions.json"
        load_concepts_from_json(concepts_path)
    else:
        # Load from local file
        load_concepts_from_json()
    
    logger.info("Service Ready!")
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("AURELIA Financial RAG Service shutting down...")


# ============================================================================
# Run with uvicorn
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
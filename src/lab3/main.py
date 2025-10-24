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
    GenerationContext,
    ConceptNote
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing AURELIA RAG Service...")


# ============================================================================
# GCS Helper Functions
# ============================================================================

def get_latest_date_from_gcs(bucket_name: str, prefix: str) -> str:
    """Auto-detect latest date folder in GCS bucket"""
    try:
        from google.cloud import storage
        from datetime import datetime
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        dates = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 2 and parts[0] == prefix.rstrip('/'):
                date_str = parts[1]
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    dates.add(date_str)
                except ValueError:
                    continue
        
        if not dates:
            raise ValueError(f"No date folders found in gs://{bucket_name}/{prefix}")
        
        latest = sorted(dates)[-1]
        logger.info(f"Auto-detected latest date: {latest}")
        return latest
        
    except Exception as e:
        logger.error(f"Failed to detect latest date: {e}")
        raise


def download_chromadb_from_gcs(
    bucket_name: str,
    date: str = None,
    local_chromadb_path: str = "/tmp/chromadb"
) -> str:
    """Download pre-built ChromaDB from GCS - Downloads chroma.sqlite3 FIRST"""
    import os
    from pathlib import Path
    from google.cloud import storage
    
    try:
        logger.info("="*70)
        logger.info("DOWNLOADING PRE-BUILT CHROMADB FROM GCS")
        logger.info("="*70)
        
        if date is None:
            date = get_latest_date_from_gcs(bucket_name, 'chromadb/')
        
        logger.info(f"Using date: {date}")
        
        os.makedirs(local_chromadb_path, exist_ok=True)
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        chromadb_prefix = f'chromadb/{date}/'
        
        # Download chroma.sqlite3 FIRST (most important!)
        logger.info("Downloading chroma.sqlite3...")
        
        sqlite_blob = bucket.blob(f'{chromadb_prefix}chroma.sqlite3')
        if sqlite_blob.exists():
            sqlite_path = Path(local_chromadb_path) / 'chroma.sqlite3'
            sqlite_blob.download_to_filename(str(sqlite_path))
            
            if sqlite_blob.size:
                size_mb = sqlite_blob.size / 1024 / 1024
                logger.info(f"  ✅ chroma.sqlite3 ({size_mb:.1f} MB)")
        else:
            raise FileNotFoundError(f"chroma.sqlite3 not found!")
        
        # Download remaining files
        logger.info("Downloading remaining files...")
        
        blobs = list(bucket.list_blobs(prefix=chromadb_prefix))
        downloaded_count = 1
        
        for blob in blobs:
            if 'chroma.sqlite3' in blob.name:
                continue
            
            if blob.name.endswith('/'):
                continue
            
            if 'manifest.json' in blob.name:
                continue
            
            relative_path = blob.name.replace(chromadb_prefix, '')
            local_file_path = Path(local_chromadb_path) / relative_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob.download_to_filename(str(local_file_path))
            downloaded_count += 1
            
            if blob.size:
                size_kb = blob.size / 1024
                logger.info(f"  ✅ {relative_path} ({size_kb:.1f} KB)")
        
        logger.info(f"\n✅ Downloaded {downloaded_count} files to {local_chromadb_path}")
        logger.info("="*70)
        
        return local_chromadb_path
        
    except Exception as e:
        logger.error(f"ChromaDB download failed: {e}")
        raise RuntimeError(f"ChromaDB download failed: {str(e)}")


# ============================================================================
# Core RAG Logic
# ============================================================================

def load_concepts_from_json(file_path_or_gcs: str = None) -> int:
    """Load pre-generated concepts into cache"""
    import json
    
    try:
        cache_service = get_cache_service()
    except Exception as e:
        logger.warning(f"Cache unavailable: {e}")
        return 0
    
    loaded_count = 0
    concepts_data = None
    
    try:
        if file_path_or_gcs is None:
            try:
                from google.cloud import storage
                
                client = storage.Client()
                bucket = client.bucket('aurelia-rag-data')
                
                date = get_latest_date_from_gcs('aurelia-rag-data', 'concepts/')
                blob = bucket.blob(f'concepts/{date}/concept_definitions.json')
                
                if blob.exists():
                    concepts_json = blob.download_as_text()
                    concepts_data = json.loads(concepts_json)
                    logger.info(f"Loaded concepts from GCS: {date}")
                else:
                    logger.info("No concepts in GCS")
                    return 0
                    
            except Exception as e:
                logger.info(f"GCS concepts not available: {e}")
                return 0
        
        elif file_path_or_gcs.startswith('gs://'):
            from google.cloud import storage
            
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
                return 0
        
        else:
            with open(file_path_or_gcs, 'r', encoding='utf-8') as f:
                concepts_data = json.load(f)
        
        if not concepts_data or not isinstance(concepts_data, list):
            return 0
        
        logger.info(f"Loading {len(concepts_data)} concepts...")
        
        for concept_dict in concepts_data:
            try:
                if 'concept_name' not in concept_dict:
                    continue
                
                concept_note = ConceptNote(**concept_dict)
                if cache_service.save(concept_note):
                    loaded_count += 1
                    
            except Exception as e:
                pass
        
        logger.info(f"✅ Loaded {loaded_count} concepts")
        return loaded_count
        
    except Exception as e:
        logger.error(f"Error loading concepts: {e}")
        return 0


def is_finance_related(concept: str) -> bool:
    """Check if concept is finance-related using OpenAI"""
    from openai import OpenAI
    
    client = OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You classify if a query is finance/economics related. Respond ONLY 'yes' or 'no'."
                },
                {
                    "role": "user",
                    "content": f"Is this finance-related? Concept: {concept}"
                }
            ],
            max_tokens=5,
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip().lower()
        return answer == 'yes'
        
    except Exception as e:
        logger.warning(f"Finance check failed: {e}")
        return True


def retrieve_context(concept: str):
    """Retrieve context from ChromaDB or Wikipedia"""
    embedding_service = get_embedding_service()
    vector_store = get_vector_store_service()
    wikipedia_service = get_wikipedia_service()
    
    query_embedding = embedding_service.embed_query(concept)
    chromadb_results = vector_store.query(query_embedding)
    
    if chromadb_results.documents and chromadb_results.similarities:
        max_similarity = max(chromadb_results.similarities)
        logger.info(f"ChromaDB similarity: {max_similarity:.3f}")
        
        if max_similarity >= settings.similarity_threshold:
            logger.info("Using ChromaDB results")
            return chromadb_results
    
    logger.info(f"No PDF match for '{concept}'")
    
    if not is_finance_related(concept):
        logger.warning(f"Rejecting non-finance: '{concept}'")
        raise HTTPException(
            status_code=400,
            detail=f"'{concept}' not found in fintbx.pdf and not finance-related"
        )
    
    logger.warning(f"Using Wikipedia for '{concept}'")
    try:
        return wikipedia_service.search(concept)
    except Exception as e:
        logger.error(f"Wikipedia failed: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"No information found for '{concept}'"
        )


def generate_note(concept: str, force_refresh: bool = False):
    """Generate concept note with caching"""
    start_time = time.time()
    cache_service = get_cache_service()
    
    if not force_refresh:
        cached_note = cache_service.get(concept)
        if cached_note:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Cache HIT: '{concept}' ({elapsed_ms:.1f}ms)")
            return cached_note, 0, elapsed_ms, True
    
    retrieval_result = retrieve_context(concept)
    
    generation_context = GenerationContext(
        concept=concept,
        context=retrieval_result.get_context(max_length=settings.max_context_tokens),
        source=retrieval_result.source,
        page_references=retrieval_result.get_page_references(),
        confidence=retrieval_result.calculate_confidence()
    )
    
    generation_service = get_generation_service()
    concept_note = generation_service.generate_concept_note(generation_context)
    
    cache_service.save(concept_note)
    
    elapsed_ms = (time.time() - start_time) * 1000
    return concept_note, len(retrieval_result.documents), elapsed_ms, False


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
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
    """Health check"""
    try:
        vector_store = get_vector_store_service()
        chromadb_healthy = vector_store.health_check()
        vector_count = vector_store.collection_size if chromadb_healthy else None
        
        generation_service = get_generation_service()
        openai_healthy = generation_service.health_check()
        
        cache_service = get_cache_service()
        cache_healthy = cache_service.health_check()
        cached_count = cache_service.count() if cache_healthy else None
        
        all_healthy = chromadb_healthy and openai_healthy and cache_healthy
        
        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            chromadb_status="connected" if chromadb_healthy else "disconnected",
            database_status="connected" if cache_healthy else "disconnected",
            openai_status="connected" if openai_healthy else "disconnected",
            vector_count=vector_count,
            cached_concepts=cached_count
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_concept(request: QueryRequest):
    """Query endpoint"""
    try:
        logger.info(f"Query: '{request.concept}', refresh={request.force_refresh}")
        
        concept_note, chunks, time_ms, cached = generate_note(
            concept=request.concept,
            force_refresh=request.force_refresh
        )
        
        return QueryResponse(
            concept_note=concept_note,
            retrieved_chunks=chunks,
            generation_time_ms=time_ms,
            cached=cached
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/seed", response_model=SeedResponse, tags=["Seed"])
async def seed_concepts(request: SeedRequest):
    """Seed endpoint"""
    try:
        logger.info(f"Seed: {len(request.concepts)} concepts")
        
        start_time = time.time()
        seeded = 0
        failed = 0
        failed_concepts = []
        
        for concept in request.concepts:
            try:
                generate_note(concept, force_refresh=True)
                seeded += 1
            except Exception as e:
                logger.error(f"Failed: '{concept}' - {e}")
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
        logger.error(f"Seed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup and Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("AURELIA Financial RAG Service Starting...")
    
    import os
    from pathlib import Path
    
    gcs_bucket = os.getenv('CHROMADB_GCS_BUCKET')
    is_cloud = gcs_bucket is not None
    
    logger.info(f"Environment: {'CLOUD' if is_cloud else 'LOCAL'}")
    
    chromadb_path = None
    
    if is_cloud:
        chromadb_path = download_chromadb_from_gcs(
            bucket_name=gcs_bucket,
            date="2025-10-24"
        )
    else:
        chromadb_path = str(settings.chromadb_path)
        
        if not Path(chromadb_path).exists():
            logger.error(f"Local ChromaDB not found: {chromadb_path}")
            raise RuntimeError(f"ChromaDB not found: {chromadb_path}")
    
    get_embedding_service()
    get_vector_store_service(chromadb_path)
    get_wikipedia_service()
    get_generation_service()
    get_cache_service()
    
    from database.models import init_database
    init_database()
    
    if is_cloud:
        try:
            load_concepts_from_json()
        except Exception as e:
            logger.warning(f"Could not load concepts: {e}")
    else:
        load_concepts_from_json()
    
    logger.info("✅ Service Ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    logger.info("AURELIA shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
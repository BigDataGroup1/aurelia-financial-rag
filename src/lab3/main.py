"""
AURELIA Financial RAG Service - Main FastAPI Application
Lab 3: FastAPI RAG Service with Retrieval, Generation, and Caching
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

from .config import settings
from .models import (
    QueryRequest, QueryResponse,
    SeedRequest, SeedResponse,
    HealthResponse,
    GenerationContext
)
from .services import (
    get_embedding_service,
    get_vector_store_service,
    get_wikipedia_service,
    get_generation_service
)
from .database.cache import get_cache_service

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


# ============================================================================
# Core RAG Logic
# ============================================================================
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
    logger.info("="*70)
    logger.info("AURELIA Financial RAG Service Starting...")
    logger.info("="*70)
    
    try:
        # Initialize services (triggers singleton creation)
        get_embedding_service()
        get_vector_store_service()
        get_wikipedia_service()
        get_generation_service()
        get_cache_service()  # ← ADD THIS
        
        # Initialize database
        from .database.models import init_database  # ← ADD THIS
        init_database()  # ← ADD THIS
        logger.info("✓ Database initialized")  # ← ADD THIS
        
        logger.info("✓ All services initialized successfully")
        logger.info(f"✓ ChromaDB: {settings.chromadb_path}")
        logger.info(f"✓ LLM Model: {settings.llm_model}")
        logger.info(f"✓ Similarity Threshold: {settings.similarity_threshold}")
        logger.info("="*70)
        logger.info("Service Ready! Listening for requests...")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

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
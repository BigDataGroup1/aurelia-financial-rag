"""
Pydantic models for API requests, responses, and structured outputs
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


# ============================================================================
# Structured Output Models (for Instructor)
# ============================================================================

class ConceptNote(BaseModel):
    """
    Structured financial concept note - enforced by Instructor
    This is the core output format for all generated concept notes
    """
    concept_name: str = Field(
        ..., 
        description="Name of the financial concept",
        min_length=1,
        max_length=255
    )
    
    definition: str = Field(
        ..., 
        description="Clear, concise definition in 2-3 sentences",
        min_length=20
    )
    
    key_components: List[str] = Field(
        ..., 
        description="3-5 key components or elements of this concept",
        min_length=3,
        max_length=5
    )
    
    formula: Optional[str] = Field(
        None, 
        description="Mathematical formula if applicable (LaTeX format preferred)"
    )
    
    example: str = Field(
        ..., 
        description="Practical example demonstrating the concept",
        min_length=50
    )
    
    use_cases: List[str] = Field(
        ..., 
        description="2-4 real-world applications or use cases",
        min_length=2,
        max_length=4
    )
    
    related_concepts: List[str] = Field(
        default_factory=list,
        description="Related financial concepts (0-5)",
        max_length=5
    )
    
    source: str = Field(
        ..., 
        description="Source of information: 'fintbx.pdf' or 'wikipedia'"
    )
    
    page_references: Optional[List[int]] = Field(
        None,
        description="Page numbers from fintbx.pdf if applicable"
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score based on retrieval quality (0.0-1.0)"
    )
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Ensure source is one of the allowed values"""
        allowed = ['fintbx.pdf', 'wikipedia']
        if v not in allowed:
            raise ValueError(f"Source must be one of {allowed}")
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Round confidence to 2 decimal places"""
        return round(v, 2)


# ============================================================================
# API Request Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    concept: str = Field(
        ...,
        description="Financial concept to query",
        min_length=2,
        max_length=200,
        examples=["Duration", "Sharpe Ratio", "Black-Scholes"]
    )
    
    force_refresh: bool = Field(
        default=False,
        description="If True, bypass cache and regenerate note"
    )
    
    @field_validator('concept')
    @classmethod
    def clean_concept_name(cls, v: str) -> str:
        """Clean and normalize concept name"""
        return v.strip()


class SeedRequest(BaseModel):
    """Request model for /seed endpoint"""
    concepts: List[str] = Field(
        ...,
        description="List of concepts to pre-generate and cache",
        min_length=1,
        max_length=100
    )
    
    @field_validator('concepts')
    @classmethod
    def clean_concepts(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate concept names"""
        return list(set(c.strip() for c in v if c.strip()))


# ============================================================================
# API Response Models
# ============================================================================

class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    concept_note: ConceptNote
    retrieved_chunks: int = Field(
        ...,
        description="Number of chunks retrieved from vector store"
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate response in milliseconds"
    )
    cached: bool = Field(
        ...,
        description="Whether result was retrieved from cache"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)"
    )


class SeedResponse(BaseModel):
    """Response model for /seed endpoint"""
    total_concepts: int = Field(
        ...,
        description="Total number of concepts requested"
    )
    seeded: int = Field(
        ...,
        description="Number successfully generated and cached"
    )
    failed: int = Field(
        ...,
        description="Number that failed to generate"
    )
    failed_concepts: List[str] = Field(
        default_factory=list,
        description="List of concepts that failed"
    )
    generation_time_ms: float = Field(
        ...,
        description="Total time taken in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str = Field(
        ...,
        description="Overall service status"
    )
    chromadb_status: str = Field(
        ...,
        description="ChromaDB connection status"
    )
    database_status: str = Field(
        ...,
        description="Cache database status"
    )
    openai_status: str = Field(
        ...,
        description="OpenAI API status"
    )
    vector_count: Optional[int] = Field(
        None,
        description="Number of vectors in ChromaDB collection"
    )
    cached_concepts: Optional[int] = Field(
        None,
        description="Number of concepts in cache"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp (UTC)"
    )


# ============================================================================
# Internal Models (for service layer)
# ============================================================================

class RetrievalResult(BaseModel):
    """Internal model for retrieval results"""
    documents: List[str]
    metadatas: List[dict]
    similarities: List[float]
    source: str  # 'chromadb' or 'wikipedia'
    
    def get_context(self, max_length: int = 6000) -> str:
        """Assemble context from documents with length limit"""
        context_parts = []
        current_length = 0
        
        for doc in self.documents:
            if current_length + len(doc) > max_length:
                break
            context_parts.append(doc)
            current_length += len(doc)
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_page_references(self) -> Optional[List[int]]:
        """Extract unique page numbers from metadata"""
        if self.source != 'fintbx.pdf':
            return None
        
        pages = set()
        for meta in self.metadatas:
            if 'page' in meta:
                pages.add(meta['page'])
        
        return sorted(list(pages)) if pages else None
    
    def calculate_confidence(self) -> float:
        """Calculate confidence score based on similarities"""
        if not self.similarities:
            return 0.5
        
        # Average of top similarities
        avg_similarity = sum(self.similarities) / len(self.similarities)
        
        # Boost if source is chromadb (primary source)
        if self.source == 'chromadb':
            return min(avg_similarity * 1.1, 1.0)
        else:
            # Wikipedia fallback gets lower confidence
            return avg_similarity * 0.8


class GenerationContext(BaseModel):
    """Context for LLM generation"""
    concept: str
    context: str
    source: str
    page_references: Optional[List[int]]
    confidence: float
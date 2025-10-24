"""
Lab 5: Evaluation Models
Pydantic schemas for evaluation metrics, results, and reports
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


# ============================================================================
# GROUND TRUTH MODELS
# ============================================================================

class GroundTruthConcept(BaseModel):
    """Reference answer for a financial concept"""
    concept_name: str
    definition: str
    key_components: List[str]
    formula: Optional[str]
    example: str
    use_cases: List[str]
    related_concepts: List[str]
    source: str
    page_references: List[int]
    confidence: float


# ============================================================================
# QUALITY METRICS
# ============================================================================

class ConceptQualityMetrics(BaseModel):
    """Quality metrics for a single concept evaluation"""
    concept: str
    
    # Accuracy (semantic similarity)
    definition_similarity: float = Field(ge=0.0, le=1.0)
    
    # Completeness (field coverage)
    has_definition: bool
    has_key_components: bool
    has_example: bool
    has_use_cases: bool
    has_formula: bool  # If applicable
    completeness_score: float = Field(ge=0.0, le=1.0)
    
    # Citation Fidelity
    page_refs_correct: int
    page_refs_total: int
    page_refs_extra: int  # Pages cited but not in ground truth
    citation_fidelity: float = Field(ge=0.0, le=1.0)
    
    # Overall Quality Score
    quality_score: float = Field(ge=0.0, le=100.0)
    
    # Weights used
    accuracy_weight: float = 0.40
    completeness_weight: float = 0.30
    citation_weight: float = 0.30


class AggregateQualityMetrics(BaseModel):
    """Aggregated quality metrics across all evaluated concepts"""
    total_concepts: int
    
    # Average scores
    avg_accuracy: float = Field(ge=0.0, le=1.0)
    avg_completeness: float = Field(ge=0.0, le=1.0)
    avg_citation_fidelity: float = Field(ge=0.0, le=1.0)
    avg_quality_score: float = Field(ge=0.0, le=100.0)
    
    # Distribution
    min_quality_score: float
    max_quality_score: float
    std_quality_score: float
    
    # Breakdown by component
    concepts_with_formulas: int
    formulas_present_rate: float


# ============================================================================
# LATENCY METRICS
# ============================================================================

class LatencyMetrics(BaseModel):
    """Latency comparison metrics"""
    
    # Cached queries
    cached_samples: int
    cached_avg_ms: float
    cached_min_ms: float
    cached_max_ms: float
    cached_std_ms: float
    
    # Fresh queries
    fresh_samples: int
    fresh_avg_ms: float
    fresh_min_ms: float
    fresh_max_ms: float
    fresh_std_ms: float
    
    # Breakdown (fresh only)
    retrieval_avg_ms: Optional[float] = None
    generation_avg_ms: Optional[float] = None
    
    # Comparison
    speedup_factor: float
    median_cached_ms: float
    median_fresh_ms: float


# ============================================================================
# COST METRICS
# ============================================================================

class TokenCostMetrics(BaseModel):
    """Token usage and cost tracking"""
    
    # Embedding costs
    embedding_model: str
    embedding_tokens: int
    embedding_cost_per_1m: float = 0.13  # text-embedding-3-large
    embedding_cost_usd: float
    
    # Generation costs
    llm_model: str
    llm_input_tokens: int
    llm_output_tokens: int
    llm_input_cost_per_1m: float = 0.150  # gpt-4o-mini
    llm_output_cost_per_1m: float = 0.600  # gpt-4o-mini
    llm_cost_usd: float
    
    # Total
    total_cost_usd: float
    cost_per_query_usd: float


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

class RetrievalPerformanceMetrics(BaseModel):
    """Retrieval performance metrics"""
    vector_store: str  # "chromadb" or "pinecone"
    
    # Latency
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    # Quality
    avg_similarity: float = Field(ge=0.0, le=1.0)
    avg_chunks_retrieved: float
    
    # Success rate
    queries_successful: int
    queries_failed: int
    success_rate: float = Field(ge=0.0, le=1.0)


# ============================================================================
# INDIVIDUAL QUERY RESULT
# ============================================================================

class QueryEvaluationResult(BaseModel):
    """Evaluation result for a single query"""
    concept: str
    
    # Response
    generated_note: Dict  # ConceptNote as dict
    cached: bool
    
    # Quality
    quality_metrics: ConceptQualityMetrics
    
    # Performance
    total_latency_ms: float
    retrieval_latency_ms: Optional[float]
    generation_latency_ms: Optional[float]
    
    # Cost
    embedding_tokens: int
    llm_input_tokens: int
    llm_output_tokens: int


# ============================================================================
# EVALUATION SUMMARY
# ============================================================================

class EvaluationSummary(BaseModel):
    """Complete evaluation summary"""
    evaluation_id: str
    timestamp: datetime
    
    # Test configuration
    total_queries: int
    cached_queries: int
    fresh_queries: int
    ground_truth_concepts: int
    
    # Quality
    quality_metrics: AggregateQualityMetrics
    
    # Performance
    latency_metrics: LatencyMetrics
    
    # Cost
    cost_metrics: TokenCostMetrics
    
    # Retrieval
    chromadb_metrics: Optional[RetrievalPerformanceMetrics] = None
    pinecone_metrics: Optional[RetrievalPerformanceMetrics] = None
    
    # GCS paths
    gcs_results_path: Optional[str] = None
    gcs_report_path: Optional[str] = None


# ============================================================================
# API MODELS
# ============================================================================

class EvaluateRequest(BaseModel):
    """Request for /evaluate endpoint"""
    test_queries: Optional[List[str]] = None  # If None, use all GT concepts
    force_refresh: bool = False
    include_latency: bool = True
    include_cost: bool = True
    include_retrieval: bool = True
    save_to_gcs: bool = True


class EvaluateResponse(BaseModel):
    """Response from /evaluate endpoint"""
    evaluation_id: str
    timestamp: datetime
    summary: EvaluationSummary
    detailed_results: List[QueryEvaluationResult]
    report_markdown: str
    gcs_path: Optional[str] = None
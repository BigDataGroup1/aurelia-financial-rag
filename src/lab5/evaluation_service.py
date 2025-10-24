"""
Lab 5: Evaluation Service
Core logic for evaluating concept note quality, latency, and costs
"""
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from openai import OpenAI

# Import from parent lab3 directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lab3"))

from config import settings
from models import ConceptNote
from services import get_embedding_service
from evaluation_models import (
    GroundTruthConcept,
    ConceptQualityMetrics,
    AggregateQualityMetrics,
    LatencyMetrics,
    TokenCostMetrics,
    RetrievalPerformanceMetrics,
    QueryEvaluationResult,
    EvaluationSummary
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for evaluating AURELIA RAG system"""
    
    def __init__(self, ground_truth_path: str = None):
        """
        Initialize evaluation service
        
        Args:
            ground_truth_path: Path to ground_truth.json
        """
        if ground_truth_path is None:
            ground_truth_path = Path(__file__).parent / "data" / "ground_truth.json"
        
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth: Dict[str, GroundTruthConcept] = {}
        self.embedding_service = get_embedding_service()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        self._load_ground_truth()
        
        logger.info(f"EvaluationService initialized with {len(self.ground_truth)} ground truth concepts")
    
    def _load_ground_truth(self):
        """Load ground truth from JSON"""
        try:
            with open(self.ground_truth_path, 'r') as f:
                gt_data = json.load(f)
            
            for item in gt_data:
                concept = GroundTruthConcept(**item)
                self.ground_truth[concept.concept_name.lower()] = concept
            
            logger.info(f"Loaded {len(self.ground_truth)} ground truth concepts")
            
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            raise
    
    # ========================================================================
    # QUALITY EVALUATION
    # ========================================================================
    
    def evaluate_quality(
        self,
        concept: str,
        generated_note: ConceptNote
    ) -> ConceptQualityMetrics:
        """
        Evaluate quality of generated concept note
        
        Args:
            concept: Concept name
            generated_note: Generated ConceptNote
            
        Returns:
            ConceptQualityMetrics
        """
        # Get ground truth
        gt = self.ground_truth.get(concept.lower())
        
        if not gt:
            raise ValueError(f"No ground truth found for '{concept}'")
        
        # 1. ACCURACY: Semantic similarity of definitions
        definition_similarity = self._compute_semantic_similarity(
            gt.definition,
            generated_note.definition
        )
        
        # 2. COMPLETENESS: Field coverage
        has_definition = bool(generated_note.definition and len(generated_note.definition) >= 20)
        has_key_components = bool(generated_note.key_components and len(generated_note.key_components) >= 3)
        has_example = bool(generated_note.example and len(generated_note.example) >= 50)
        has_use_cases = bool(generated_note.use_cases and len(generated_note.use_cases) >= 2)
        
        # Formula is optional
        gt_has_formula = gt.formula is not None
        has_formula = generated_note.formula is not None if gt_has_formula else True
        
        # Completeness score
        completeness_fields = [
            has_definition,
            has_key_components,
            has_example,
            has_use_cases,
            has_formula
        ]
        completeness_score = sum(completeness_fields) / len(completeness_fields)
        
        # 3. CITATION FIDELITY: Page reference accuracy
        citation_fidelity, page_refs_correct, page_refs_total, page_refs_extra = \
            self._evaluate_citations(gt, generated_note)
        
        # 4. OVERALL QUALITY SCORE (0-100)
        accuracy_weight = 0.40
        completeness_weight = 0.30
        citation_weight = 0.30
        
        quality_score = (
            accuracy_weight * definition_similarity * 100 +
            completeness_weight * completeness_score * 100 +
            citation_weight * citation_fidelity * 100
        )
        
        return ConceptQualityMetrics(
            concept=concept,
            definition_similarity=definition_similarity,
            has_definition=has_definition,
            has_key_components=has_key_components,
            has_example=has_example,
            has_use_cases=has_use_cases,
            has_formula=has_formula,
            completeness_score=completeness_score,
            page_refs_correct=page_refs_correct,
            page_refs_total=page_refs_total,
            page_refs_extra=page_refs_extra,
            citation_fidelity=citation_fidelity,
            quality_score=quality_score,
            accuracy_weight=accuracy_weight,
            completeness_weight=completeness_weight,
            citation_weight=citation_weight
        )
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using embeddings
        
        Returns:
            Cosine similarity (0-1)
        """
        try:
            # Generate embeddings
            emb1 = self.embedding_service.embed_query(text1)
            emb2 = self.embedding_service.embed_query(text2)
            
            # Cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            similarity = dot_product / (norm1 * norm2)
            
            # Clip to [0, 1]
            similarity = np.clip(similarity, 0.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.5  # Default neutral score
    
    def _evaluate_citations(
        self,
        gt: GroundTruthConcept,
        generated: ConceptNote
    ) -> Tuple[float, int, int, int]:
        """
        Evaluate citation accuracy
        
        Returns:
            (citation_fidelity, correct_refs, total_refs, extra_refs)
        """
        if generated.source != "fintbx.pdf":
            # Wikipedia fallback - no page refs expected
            return 0.7, 0, 0, 0  # Fixed score for Wikipedia
        
        gt_pages = set(gt.page_references)
        gen_pages = set(generated.page_references or [])
        
        # Correct pages
        correct_pages = gt_pages.intersection(gen_pages)
        page_refs_correct = len(correct_pages)
        
        # Total expected pages
        page_refs_total = len(gt_pages)
        
        # Extra pages (cited but not in GT)
        extra_pages = gen_pages - gt_pages
        page_refs_extra = len(extra_pages)
        
        # Citation fidelity score
        if page_refs_total == 0:
            citation_fidelity = 1.0
        else:
            # Reward correct citations, penalize missing and extra
            precision = page_refs_correct / len(gen_pages) if gen_pages else 0.0
            recall = page_refs_correct / page_refs_total
            
            # F1 score
            if precision + recall > 0:
                citation_fidelity = 2 * (precision * recall) / (precision + recall)
            else:
                citation_fidelity = 0.0
        
        return citation_fidelity, page_refs_correct, page_refs_total, page_refs_extra
    
    # ========================================================================
    # TOKEN COST ESTIMATION
    # ========================================================================
    
    def estimate_token_cost(
        self,
        query_results: List[QueryEvaluationResult]
    ) -> TokenCostMetrics:
        """
        Estimate token costs for all queries
        
        Args:
            query_results: List of query evaluation results
            
        Returns:
            TokenCostMetrics
        """
        # Sum tokens across all queries
        total_embedding_tokens = sum(r.embedding_tokens for r in query_results)
        total_llm_input_tokens = sum(r.llm_input_tokens for r in query_results)
        total_llm_output_tokens = sum(r.llm_output_tokens for r in query_results)
        
        # Costs (text-embedding-3-large: $0.13/1M, gpt-4o-mini: $0.15/1M input, $0.60/1M output)
        embedding_cost = (total_embedding_tokens / 1_000_000) * 0.13
        llm_cost = (
            (total_llm_input_tokens / 1_000_000) * 0.150 +
            (total_llm_output_tokens / 1_000_000) * 0.600
        )
        
        total_cost = embedding_cost + llm_cost
        cost_per_query = total_cost / len(query_results) if query_results else 0.0
        
        return TokenCostMetrics(
            embedding_model="text-embedding-3-large",
            embedding_tokens=total_embedding_tokens,
            embedding_cost_usd=embedding_cost,
            llm_model="gpt-4o-mini",
            llm_input_tokens=total_llm_input_tokens,
            llm_output_tokens=total_llm_output_tokens,
            llm_cost_usd=llm_cost,
            total_cost_usd=total_cost,
            cost_per_query_usd=cost_per_query
        )
    
    # ========================================================================
    # AGGREGATE METRICS
    # ========================================================================
    
    def aggregate_quality_metrics(
        self,
        query_results: List[QueryEvaluationResult]
    ) -> AggregateQualityMetrics:
        """Aggregate quality metrics across all queries"""
        
        quality_scores = [r.quality_metrics.quality_score for r in query_results]
        accuracies = [r.quality_metrics.definition_similarity for r in query_results]
        completenesses = [r.quality_metrics.completeness_score for r in query_results]
        citations = [r.quality_metrics.citation_fidelity for r in query_results]
        
        concepts_with_formulas = sum(
            1 for r in query_results 
            if r.quality_metrics.has_formula
        )
        
        return AggregateQualityMetrics(
            total_concepts=len(query_results),
            avg_accuracy=float(np.mean(accuracies)),
            avg_completeness=float(np.mean(completenesses)),
            avg_citation_fidelity=float(np.mean(citations)),
            avg_quality_score=float(np.mean(quality_scores)),
            min_quality_score=float(np.min(quality_scores)),
            max_quality_score=float(np.max(quality_scores)),
            std_quality_score=float(np.std(quality_scores)),
            concepts_with_formulas=concepts_with_formulas,
            formulas_present_rate=concepts_with_formulas / len(query_results)
        )
    
    def aggregate_latency_metrics(
        self,
        query_results: List[QueryEvaluationResult]
    ) -> LatencyMetrics:
        """Aggregate latency metrics"""
        
        cached_results = [r for r in query_results if r.cached]
        fresh_results = [r for r in query_results if not r.cached]
        
        # Cached latencies
        if cached_results:
            cached_latencies = [r.total_latency_ms for r in cached_results]
            cached_avg = float(np.mean(cached_latencies))
            cached_min = float(np.min(cached_latencies))
            cached_max = float(np.max(cached_latencies))
            cached_std = float(np.std(cached_latencies))
            cached_median = float(np.median(cached_latencies))
        else:
            cached_avg = cached_min = cached_max = cached_std = cached_median = 0.0
        
        # Fresh latencies
        if fresh_results:
            fresh_latencies = [r.total_latency_ms for r in fresh_results]
            fresh_avg = float(np.mean(fresh_latencies))
            fresh_min = float(np.min(fresh_latencies))
            fresh_max = float(np.max(fresh_latencies))
            fresh_std = float(np.std(fresh_latencies))
            fresh_median = float(np.median(fresh_latencies))
            
            # Breakdown
            retrieval_latencies = [r.retrieval_latency_ms for r in fresh_results if r.retrieval_latency_ms]
            generation_latencies = [r.generation_latency_ms for r in fresh_results if r.generation_latency_ms]
            
            retrieval_avg = float(np.mean(retrieval_latencies)) if retrieval_latencies else None
            generation_avg = float(np.mean(generation_latencies)) if generation_latencies else None
        else:
            fresh_avg = fresh_min = fresh_max = fresh_std = fresh_median = 0.0
            retrieval_avg = generation_avg = None
        
        # Speedup factor
        speedup = fresh_avg / cached_avg if cached_avg > 0 else 1.0
        
        return LatencyMetrics(
            cached_samples=len(cached_results),
            cached_avg_ms=cached_avg,
            cached_min_ms=cached_min,
            cached_max_ms=cached_max,
            cached_std_ms=cached_std,
            fresh_samples=len(fresh_results),
            fresh_avg_ms=fresh_avg,
            fresh_min_ms=fresh_min,
            fresh_max_ms=fresh_max,
            fresh_std_ms=fresh_std,
            retrieval_avg_ms=retrieval_avg,
            generation_avg_ms=generation_avg,
            speedup_factor=speedup,
            median_cached_ms=cached_median,
            median_fresh_ms=fresh_median
        )
    
    # ========================================================================
    # TOKEN ESTIMATION
    # ========================================================================
    
    def estimate_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """
        Estimate token count for text
        
        Rough estimation: ~4 characters per token
        """
        return len(text) // 4
    
    # ========================================================================
    # MAIN EVALUATION FUNCTION
    # ========================================================================
    
    def evaluate_query(
        self,
        concept: str,
        generated_note: ConceptNote,
        cached: bool,
        total_latency_ms: float,
        retrieval_latency_ms: Optional[float] = None,
        generation_latency_ms: Optional[float] = None,
        retrieved_chunks: int = 0
    ) -> QueryEvaluationResult:
        """
        Evaluate a single query result
        """
        # Quality evaluation
        quality_metrics = self.evaluate_quality(concept, generated_note)
        
        # Token estimation
        if not cached:
            # Embedding tokens (query only)
            embedding_tokens = self.estimate_tokens(concept)
            
            # LLM tokens
            # Input: context + prompt
            context_size = retrieved_chunks * 400  # ~400 chars per chunk
            llm_input_tokens = self.estimate_tokens(
                concept + generated_note.definition + str(context_size)
            )
            
            # Output: generated concept note - FIXED: model_dump() instead of dict()
            output_text = json.dumps(generated_note.model_dump())
            llm_output_tokens = self.estimate_tokens(output_text)
        else:
            embedding_tokens = 0
            llm_input_tokens = 0
            llm_output_tokens = 0
        
        # FIXED: model_dump() instead of dict()
        return QueryEvaluationResult(
            concept=concept,
            generated_note=generated_note.model_dump(),
            cached=cached,
            quality_metrics=quality_metrics,
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            embedding_tokens=embedding_tokens,
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens
        )
    
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    
    def generate_markdown_report(
        self,
        summary: EvaluationSummary,
        detailed_results: List[QueryEvaluationResult]
    ) -> str:
        """Generate markdown evaluation report"""
        
        lines = [
            "# AURELIA Lab 5 - Evaluation Report",
            f"**Evaluation ID:** `{summary.evaluation_id}`",
            f"**Timestamp:** {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"**Total Queries:** {summary.total_queries}",
            f"**Cached Queries:** {summary.cached_queries}",
            f"**Fresh Queries:** {summary.fresh_queries}",
            f"**Ground Truth Concepts:** {summary.ground_truth_concepts}",
            "",
            "### Quality Metrics",
            "",
            f"- **Overall Quality Score:** {summary.quality_metrics.avg_quality_score:.1f}/100",
            f"- **Accuracy (Semantic Similarity):** {summary.quality_metrics.avg_accuracy:.2%}",
            f"- **Completeness (Field Coverage):** {summary.quality_metrics.avg_completeness:.2%}",
            f"- **Citation Fidelity:** {summary.quality_metrics.avg_citation_fidelity:.2%}",
            "",
            "### Performance Metrics",
            "",
            f"- **Cached Avg Latency:** {summary.latency_metrics.cached_avg_ms:.1f}ms",
            f"- **Fresh Avg Latency:** {summary.latency_metrics.fresh_avg_ms:.1f}ms",
            f"- **Speedup Factor:** {summary.latency_metrics.speedup_factor:.1f}x",
            "",
            "### Cost Analysis",
            "",
            f"- **Total Cost:** ${summary.cost_metrics.total_cost_usd:.4f}",
            f"- **Cost per Query:** ${summary.cost_metrics.cost_per_query_usd:.4f}",
            f"- **Embedding Tokens:** {summary.cost_metrics.embedding_tokens:,}",
            f"- **LLM Tokens:** {summary.cost_metrics.llm_input_tokens + summary.cost_metrics.llm_output_tokens:,}",
            "",
            "---",
            "",
            "## Detailed Results",
            "",
            "| Concept | Quality | Accuracy | Complete | Citation | Latency | Cached |",
            "|---------|---------|----------|----------|----------|---------|--------|"
        ]
        
        # Sort by quality score descending
        sorted_results = sorted(
            detailed_results,
            key=lambda r: r.quality_metrics.quality_score,
            reverse=True
        )
        
        for result in sorted_results:
            qm = result.quality_metrics
            cached_icon = "✓" if result.cached else "✗"
            
            lines.append(
                f"| {result.concept} | "
                f"{qm.quality_score:.1f} | "
                f"{qm.definition_similarity:.2f} | "
                f"{qm.completeness_score:.2f} | "
                f"{qm.citation_fidelity:.2f} | "
                f"{result.total_latency_ms:.0f}ms | "
                f"{cached_icon} |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            f"*Report generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*"
        ])
        
        return "\n".join(lines)


# ============================================================================
# SINGLETON
# ============================================================================

_evaluation_service = None

def get_evaluation_service(ground_truth_path: str = None) -> EvaluationService:
    """Get or create evaluation service instance"""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService(ground_truth_path)
    return _evaluation_service
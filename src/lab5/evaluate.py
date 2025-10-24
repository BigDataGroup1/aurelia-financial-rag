"""
Lab 5: Main Evaluation Script
Run evaluations on AURELIA RAG system and generate reports

Usage:
    python src/lab5/evaluate.py [--concepts CONCEPT1 CONCEPT2 ...] [--force-refresh] [--no-gcs]
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lab3"))

from config import settings
from models import ConceptNote
from main import generate_note
from evaluation_service import get_evaluation_service
from evaluation_models import (
    QueryEvaluationResult,
    EvaluationSummary
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runner for Lab 5 evaluation"""
    
    def __init__(self, save_to_gcs: bool = True):
        self.eval_service = get_evaluation_service()
        self.save_to_gcs = save_to_gcs
        
        # Create local output directory
        self.output_dir = Path(__file__).parent / "evaluation_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation ID
        self.eval_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.eval_dir = self.output_dir / self.eval_id
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluation ID: {self.eval_id}")
        logger.info(f"Output directory: {self.eval_dir}")
    
    def run_evaluation(
        self,
        test_concepts: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> EvaluationSummary:
        """
        Run full evaluation
        
        Args:
            test_concepts: List of concepts to test (None = all GT concepts)
            force_refresh: Force regeneration of cached notes
            
        Returns:
            EvaluationSummary
        """
        print("=" * 70)
        print("AURELIA LAB 5 - EVALUATION & BENCHMARKING")
        print("=" * 70)
        print()
        
        # Determine test concepts
        if test_concepts is None:
            test_concepts = list(self.eval_service.ground_truth.keys())
            print(f"📋 Testing all {len(test_concepts)} ground truth concepts")
        else:
            # Normalize concept names
            test_concepts = [c.lower() for c in test_concepts]
            print(f"📋 Testing {len(test_concepts)} specified concepts")
        
        print(f"🔄 Force refresh: {force_refresh}")
        print()
        print("-" * 70)
        print("Running Evaluations...")
        print("-" * 70)
        print()
        
        # Run queries
        query_results = []
        cached_count = 0
        fresh_count = 0
        
        for i, concept in enumerate(test_concepts, 1):
            try:
                # Get proper concept name from GT
                gt_concept = self.eval_service.ground_truth[concept]
                proper_name = gt_concept.concept_name
                
                # Time the query
                start_time = time.time()
                
                # Generate note
                concept_note, retrieved_chunks, generation_time_ms, cached = generate_note(
                    concept=proper_name,
                    force_refresh=force_refresh
                )
                
                total_latency_ms = (time.time() - start_time) * 1000
                
                # Estimate component latencies for fresh queries
                if not cached:
                    retrieval_latency_ms = 50.0  # Estimate (ChromaDB overhead)
                    generation_latency_ms = generation_time_ms
                else:
                    retrieval_latency_ms = None
                    generation_latency_ms = None
                
                # Evaluate
                result = self.eval_service.evaluate_query(
                    concept=proper_name,
                    generated_note=concept_note,
                    cached=cached,
                    total_latency_ms=total_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    retrieved_chunks=retrieved_chunks
                )
                
                query_results.append(result)
                
                # Update counts
                if cached:
                    cached_count += 1
                else:
                    fresh_count += 1
                
                # Progress
                status = "CACHED" if cached else "FRESH"
                quality = result.quality_metrics.quality_score
                
                print(
                    f"✓ Query {i}/{len(test_concepts)}: {proper_name} "
                    f"[{status}] {total_latency_ms:.0f}ms | "
                    f"Quality: {quality:.1f}/100"
                )
                
            except Exception as e:
                logger.error(f"Failed to evaluate '{concept}': {e}")
                print(f"✗ Query {i}/{len(test_concepts)}: {concept} - FAILED ({e})")
        
        print()
        print("=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print()
        
        # Aggregate metrics
        quality_metrics = self.eval_service.aggregate_quality_metrics(query_results)
        latency_metrics = self.eval_service.aggregate_latency_metrics(query_results)
        cost_metrics = self.eval_service.estimate_token_cost(query_results)
        
        # Create summary
        summary = EvaluationSummary(
            evaluation_id=self.eval_id,
            timestamp=datetime.utcnow(),
            total_queries=len(query_results),
            cached_queries=cached_count,
            fresh_queries=fresh_count,
            ground_truth_concepts=len(self.eval_service.ground_truth),
            quality_metrics=quality_metrics,
            latency_metrics=latency_metrics,
            cost_metrics=cost_metrics
        )
        
        # Print summary
        self._print_summary(summary)
        
        # Save results
        self._save_results(summary, query_results)
        
        # Upload to GCS if enabled
        if self.save_to_gcs:
            try:
                gcs_path = self._upload_to_gcs(summary, query_results)
                summary.gcs_results_path = gcs_path
                print(f"☁️  Uploaded to GCS: {gcs_path}")
            except Exception as e:
                logger.warning(f"GCS upload failed: {e}")
                print(f"⚠️  GCS upload skipped: {e}")
        
        print()
        print("=" * 70)
        
        return summary
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary to console"""
        
        qm = summary.quality_metrics
        lm = summary.latency_metrics
        cm = summary.cost_metrics
        
        print("📊 CONCEPT NOTE QUALITY:", f"{qm.avg_quality_score:.1f}/100")
        print(f"   ├─ Accuracy: {qm.avg_accuracy:.2%} (semantic similarity)")
        print(f"   ├─ Completeness: {qm.avg_completeness:.2%} (field coverage)")
        print(f"   └─ Citation Fidelity: {qm.avg_citation_fidelity:.2%} (page refs)")
        print()
        
        print("⚡ LATENCY ANALYSIS:")
        print(f"   ├─ Cached: {lm.cached_avg_ms:.1f}ms avg ({lm.cached_samples} samples)")
        print(f"   ├─ Fresh: {lm.fresh_avg_ms:.1f}ms avg ({lm.fresh_samples} samples)")
        print(f"   └─ Speedup: {lm.speedup_factor:.1f}x")
        print()
        
        print("💰 COST ANALYSIS:")
        print(f"   ├─ Embeddings: ${cm.embedding_cost_usd:.4f} ({cm.embedding_tokens:,} tokens)")
        print(f"   ├─ Generation: ${cm.llm_cost_usd:.4f} ({cm.llm_input_tokens + cm.llm_output_tokens:,} tokens)")
        print(f"   └─ Total: ${cm.total_cost_usd:.4f}")
        print()
    
    def _save_results(
        self,
        summary: EvaluationSummary,
        detailed_results: List[QueryEvaluationResult]
    ):
        """Save evaluation results locally"""
        
        # Save summary JSON - FIXED: model_dump() instead of dict()
        summary_path = self.eval_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        
        # Save detailed results JSON - FIXED: model_dump() instead of dict()
        results_path = self.eval_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(
                [r.model_dump() for r in detailed_results],
                f,
                indent=2,
                default=str
            )
        
        # Generate and save markdown report
        report = self.eval_service.generate_markdown_report(summary, detailed_results)
        report_path = self.eval_dir / "evaluation_report.md"
        report_path.write_text(report)
        
        print("✓ Results saved:")
        print(f"   - Summary: {summary_path}")
        print(f"   - Detailed: {results_path}")
        print(f"   - Report: {report_path}")
    
    def _upload_to_gcs(
        self,
        summary: EvaluationSummary,
        detailed_results: List[QueryEvaluationResult]
    ) -> str:
        """Upload results to GCS"""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(settings.gcs_bucket)
            
            # GCS path: evaluations/YYYY-MM-DD/eval_id/
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
            gcs_prefix = f"evaluations/{date_str}/{self.eval_id}"
            
            # Upload summary - FIXED: model_dump() instead of dict()
            blob = bucket.blob(f"{gcs_prefix}/summary.json")
            blob.upload_from_string(
                json.dumps(summary.model_dump(), indent=2, default=str),
                content_type='application/json'
            )
            
            # Upload detailed results - FIXED: model_dump() instead of dict()
            blob = bucket.blob(f"{gcs_prefix}/detailed_results.json")
            blob.upload_from_string(
                json.dumps([r.model_dump() for r in detailed_results], indent=2, default=str),
                content_type='application/json'
            )
            
            # Upload report
            report = self.eval_service.generate_markdown_report(summary, detailed_results)
            blob = bucket.blob(f"{gcs_prefix}/evaluation_report.md")
            blob.upload_from_string(report, content_type='text/markdown')
            
            gcs_path = f"gs://{settings.gcs_bucket}/{gcs_prefix}/"
            logger.info(f"Uploaded to GCS: {gcs_path}")
            
            return gcs_path
            
        except Exception as e:
            logger.error(f"GCS upload failed: {e}")
            raise


def main():
    """Main evaluation entry point"""
    
    parser = argparse.ArgumentParser(
        description="AURELIA Lab 5 - Evaluation & Benchmarking"
    )
    parser.add_argument(
        '--concepts',
        nargs='+',
        help='Specific concepts to evaluate (default: all GT concepts)'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force regeneration of cached notes'
    )
    parser.add_argument(
        '--no-gcs',
        action='store_true',
        help='Skip GCS upload'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(save_to_gcs=not args.no_gcs)
    
    # Run evaluation
    try:
        summary = runner.run_evaluation(
            test_concepts=args.concepts,
            force_refresh=args.force_refresh
        )
        
        print()
        print("✅ Evaluation completed successfully!")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
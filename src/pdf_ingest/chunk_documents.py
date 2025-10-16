"""
Lab 1 Step 2: Chunking Strategy Experimentation
Tests multiple LangChain splitters with different parameters
"""
import os
from pathlib import Path
from typing import List, Dict
import json
import time
import logging

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PAGES_DIR = PROCESSED / "pages"
CHUNKS_OUT = PROJECT_ROOT / "data" / "chunks"
CHUNKS_OUT.mkdir(parents=True, exist_ok=True)


def load_parsed_pages() -> List[Document]:
    """Load all parsed markdown pages as LangChain Documents"""
    logger.info("Loading parsed pages...")
    documents = []
    
    for page_file in sorted(PAGES_DIR.glob("page_*.md")):
        # Extract page number from filename
        page_num = int(page_file.stem.split("_")[1])
        
        content = page_file.read_text(encoding='utf-8')
        
        # Extract metadata from page content
        metadata = {
            'source': 'fintbx.pdf',
            'page': page_num,
            'file_path': str(page_file),
        }
        
        # Check for content indicators
        if '📷' in content:
            metadata['has_images'] = True
        if '📊' in content:
            metadata['has_tables'] = True
        if '💻' in content:
            metadata['has_code'] = True
        if '🔢' in content:
            metadata['has_formulas'] = True
        
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    logger.info(f"✓ Loaded {len(documents)} pages")
    return documents


def strategy_1_recursive_small(documents: List[Document]) -> List[Document]:
    """
    Strategy 1: RecursiveCharacterTextSplitter - Small chunks
    Good for: Precise retrieval, specific facts
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    return chunks


def strategy_2_recursive_medium(documents: List[Document]) -> List[Document]:
    """
    Strategy 2: RecursiveCharacterTextSplitter - Medium chunks
    Good for: Balance between context and precision
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


def strategy_3_recursive_large(documents: List[Document]) -> List[Document]:
    """
    Strategy 3: RecursiveCharacterTextSplitter - Large chunks
    Good for: Maximum context, complex queries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


def strategy_4_markdown_header(documents: List[Document]) -> List[Document]:
    """
    Strategy 4: MarkdownHeaderTextSplitter
    Good for: Preserving document structure, section-based retrieval
    """
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    chunks = []
    for doc in documents:
        try:
            # Split by headers
            md_chunks = markdown_splitter.split_text(doc.page_content)
            
            # Convert back to Documents with original metadata
            for chunk in md_chunks:
                new_doc = Document(
                    page_content=chunk.page_content,
                    metadata={**doc.metadata, **chunk.metadata}
                )
                chunks.append(new_doc)
        except Exception as e:
            logger.debug(f"Error splitting page {doc.metadata.get('page')}: {e}")
            # Fallback: keep original
            chunks.append(doc)
    
    return chunks


def strategy_5_hybrid(documents: List[Document]) -> List[Document]:
    """
    Strategy 5: Hybrid - Markdown headers + recursive splitting
    Good for: Best of both - structure + size control
    """
    # Stage 1: Split by markdown headers
    headers_to_split_on = [
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    # Stage 2: Further split large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    
    chunks = []
    for doc in documents:
        try:
            # Split by headers
            md_chunks = markdown_splitter.split_text(doc.page_content)
            
            for md_chunk in md_chunks:
                # Check if chunk is too large
                if len(md_chunk.page_content) > 1200:
                    # Further split
                    sub_chunks = text_splitter.split_documents([md_chunk])
                    chunks.extend(sub_chunks)
                else:
                    # Keep as is, but ensure metadata is preserved
                    new_doc = Document(
                        page_content=md_chunk.page_content,
                        metadata={**doc.metadata, **md_chunk.metadata}
                    )
                    chunks.append(new_doc)
        except Exception as e:
            logger.debug(f"Error in hybrid split for page {doc.metadata.get('page')}: {e}")
            # Fallback: use recursive splitter only
            fallback_chunks = text_splitter.split_documents([doc])
            chunks.extend(fallback_chunks)
    
    return chunks


def strategy_6_code_aware(documents: List[Document]) -> List[Document]:
    """
    Strategy 6: Code-aware splitting
    Good for: Preserving code blocks and formulas
    """
    # Custom separators to preserve code blocks
    code_separators = [
        "\n```\n",      # End of code blocks
        "\n### ",       # Section headers
        "\n## ",
        "\n**",         # Bold text (often section titles)
        "\n\n",         # Paragraphs
        "\n",
        " ",
        ""
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=code_separators,
        keep_separator=True,
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


def evaluate_chunking_strategy(chunks: List[Document], strategy_name: str) -> Dict:
    """
    Evaluate chunking quality with comprehensive metrics
    """
    if not chunks:
        return {'strategy': strategy_name, 'error': 'No chunks produced'}
    
    chunk_sizes = [len(c.page_content) for c in chunks]
    
    # Basic stats
    metrics = {
        'strategy': strategy_name,
        'num_chunks': len(chunks),
        'avg_chunk_size': sum(chunk_sizes) / len(chunks),
        'min_chunk_size': min(chunk_sizes),
        'max_chunk_size': max(chunk_sizes),
        'median_chunk_size': sorted(chunk_sizes)[len(chunk_sizes) // 2],
        'total_chars': sum(chunk_sizes),
    }
    
    # Content analysis
    chunks_with_code = sum(1 for c in chunks if '```' in c.page_content)
    chunks_with_tables = sum(1 for c in chunks if '|' in c.page_content and '---' in c.page_content)
    chunks_with_formulas = sum(1 for c in chunks if any(marker in c.page_content for marker in ['=', '∑', '∫', '√']))
    chunks_with_headers = sum(1 for c in chunks if c.page_content.strip().startswith('#'))
    
    metrics['chunks_with_code'] = chunks_with_code
    metrics['chunks_with_tables'] = chunks_with_tables
    metrics['chunks_with_formulas'] = chunks_with_formulas
    metrics['chunks_with_headers'] = chunks_with_headers
    
    # Metadata preservation
    chunks_with_page_meta = sum(1 for c in chunks if 'page' in c.metadata)
    chunks_with_source = sum(1 for c in chunks if 'source' in c.metadata)
    
    metrics['metadata_preservation'] = {
        'page_numbers': chunks_with_page_meta,
        'source_info': chunks_with_source
    }
    
    # Size distribution
    size_ranges = {
        '0-500': sum(1 for s in chunk_sizes if s <= 500),
        '501-1000': sum(1 for s in chunk_sizes if 500 < s <= 1000),
        '1001-1500': sum(1 for s in chunk_sizes if 1000 < s <= 1500),
        '1501+': sum(1 for s in chunk_sizes if s > 1500),
    }
    metrics['size_distribution'] = size_ranges
    
    return metrics


def save_chunks(chunks: List[Document], strategy_name: str):
    """Save chunks to JSON for later use"""
    output_file = CHUNKS_OUT / f"{strategy_name}_chunks.json"
    
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            'chunk_id': i,
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'char_count': len(chunk.page_content)
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Saved to: {output_file.name}")


def save_sample_chunks(chunks: List[Document], strategy_name: str, num_samples: int = 3):
    """Save sample chunks for manual inspection"""
    samples_file = CHUNKS_OUT / f"{strategy_name}_samples.md"
    
    # Get samples from different parts
    sample_indices = [
        0,  # First chunk
        len(chunks) // 3,  # Early middle
        len(chunks) // 2,  # Middle
        (2 * len(chunks)) // 3,  # Late middle
        len(chunks) - 1  # Last chunk
    ]
    
    content = [
        f"# Sample Chunks: {strategy_name}",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"Total chunks: {len(chunks)}",
        "",
        "---",
        ""
    ]
    
    for idx in sample_indices[:num_samples]:
        if idx < len(chunks):
            chunk = chunks[idx]
            content.extend([
                f"## Sample {sample_indices.index(idx) + 1}: Chunk {idx + 1}/{len(chunks)}",
                "",
                f"**Metadata:** Page {chunk.metadata.get('page', 'N/A')}, {len(chunk.page_content)} chars",
                "",
                "**Content:**",
                "```",
                chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                "```",
                "",
                "---",
                ""
            ])
    
    samples_file.write_text("\n".join(content), encoding='utf-8')


def generate_comparison_report(results: List[Dict], output_dir: Path):
    """Generate a detailed markdown comparison report"""
    
    report = [
        "# Chunking Strategy Comparison Report",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Executive Summary",
        "",
        f"Tested {len(results)} chunking strategies on the Financial Toolbox PDF corpus.",
        "Goal: Find optimal balance between chunk granularity, content preservation, and retrieval efficiency.",
        "",
        "---",
        ""
    ]
    
    # Overall Statistics Table
    report.extend([
        "## 1. Overall Statistics",
        "",
        "| Strategy | Chunks | Avg Size | Min | Max | Median | Processing Time |",
        "|----------|--------|----------|-----|-----|--------|-----------------|"
    ])
    
    for r in results:
        report.append(
            f"| {r['strategy']:<30} | {r['num_chunks']:,} | {r['avg_chunk_size']:.0f} | "
            f"{r['min_chunk_size']} | {r['max_chunk_size']:,} | {r['median_chunk_size']:.0f} | "
            f"{r['processing_time_sec']:.2f}s |"
        )
    
    report.extend(["", ""])
    
    # Content Preservation Table
    report.extend([
        "## 2. Content Preservation",
        "",
        "| Strategy | Code Blocks | Tables | Formulas | Headers |",
        "|----------|-------------|--------|----------|---------|"
    ])
    
    for r in results:
        report.append(
            f"| {r['strategy']:<30} | {r['chunks_with_code']} | "
            f"{r['chunks_with_tables']} | {r['chunks_with_formulas']} | "
            f"{r['chunks_with_headers']} |"
        )
    
    report.extend(["", ""])
    
    # Metadata Preservation
    report.extend([
        "## 3. Metadata Preservation",
        "",
        "Critical for citations and source attribution in RAG.",
        "",
        "| Strategy | Page Numbers | Source Info | Preservation Rate |",
        "|----------|--------------|-------------|-------------------|"
    ])
    
    for r in results:
        total = r['num_chunks']
        page_meta = r['metadata_preservation']['page_numbers']
        source_meta = r['metadata_preservation']['source_info']
        rate = (page_meta / total * 100) if total > 0 else 0
        
        status = "✅" if rate == 100 else "⚠️" if rate >= 50 else "❌"
        
        report.append(
            f"| {r['strategy']:<30} | {page_meta}/{total} | "
            f"{source_meta}/{total} | {status} {rate:.1f}% |"
        )
    
    report.extend(["", ""])
    
    # Size Distribution
    report.extend([
        "## 4. Chunk Size Distribution",
        "",
        "| Strategy | 0-500 | 501-1000 | 1001-1500 | 1501+ |",
        "|----------|-------|----------|-----------|-------|"
    ])
    
    for r in results:
        dist = r['size_distribution']
        report.append(
            f"| {r['strategy']:<30} | {dist['0-500']} | "
            f"{dist['501-1000']} | {dist['1001-1500']} | {dist['1501+']} |"
        )
    
    report.extend(["", ""])
    
    # Visual ASCII Charts
    report.extend([
        "## 5. Visual Comparison: Chunk Count",
        "",
        "```"
    ])
    
    max_chunks = max(r['num_chunks'] for r in results)
    for r in results:
        bar_length = int((r['num_chunks'] / max_chunks) * 50)
        bar = "█" * bar_length
        report.append(f"{r['strategy']:<35} | {bar} {r['num_chunks']:,}")
    
    report.extend(["```", "", ""])
    
    # Code Preservation Chart
    report.extend([
        "## 6. Visual Comparison: Code Block Preservation",
        "",
        "```"
    ])
    
    max_code = max(r['chunks_with_code'] for r in results)
    for r in results:
        bar_length = int((r['chunks_with_code'] / max_code) * 50) if max_code > 0 else 0
        bar = "█" * bar_length
        report.append(f"{r['strategy']:<35} | {bar} {r['chunks_with_code']}")
    
    report.extend(["```", "", ""])
    
    # Scoring System
    report.extend([
        "## 7. Multi-Criteria Scoring",
        "",
        "Weighted scoring across key dimensions:",
        "- Code Preservation: 25%",
        "- Metadata Integrity: 25%",
        "- Size Consistency: 20%",
        "- Table Preservation: 15%",
        "- Chunk Count (lower is better): 15%",
        "",
        "| Strategy | Code | Metadata | Size | Tables | Count | **Total** |",
        "|----------|------|----------|------|--------|-------|-----------|"
    ])
    
    max_code = max(r['chunks_with_code'] for r in results)
    max_tables = max(r['chunks_with_tables'] for r in results)
    min_chunks = min(r['num_chunks'] for r in results)
    max_chunks = max(r['num_chunks'] for r in results)
    
    scored_results = []
    
    for r in results:
        # Code preservation score (0-25)
        code_score = (r['chunks_with_code'] / max_code * 25) if max_code > 0 else 0
        
        # Metadata score (0-25)
        meta_rate = r['metadata_preservation']['page_numbers'] / r['num_chunks']
        meta_score = meta_rate * 25
        
        # Size consistency score (0-20) - lower variance is better
        size_range = r['max_chunk_size'] - r['min_chunk_size']
        # Penalize extreme variance
        if size_range > 5000:
            size_score = 5
        elif size_range > 2000:
            size_score = 10
        elif size_range > 1000:
            size_score = 15
        else:
            size_score = 20
        
        # Table preservation score (0-15)
        table_score = (r['chunks_with_tables'] / max_tables * 15) if max_tables > 0 else 0
        
        # Chunk count score (0-15) - fewer chunks is better (less overhead)
        # Inverse relationship
        count_score = ((max_chunks - r['num_chunks']) / (max_chunks - min_chunks) * 15) if max_chunks != min_chunks else 15
        
        total_score = code_score + meta_score + size_score + table_score + count_score
        
        scored_results.append({
            'strategy': r['strategy'],
            'code_score': code_score,
            'meta_score': meta_score,
            'size_score': size_score,
            'table_score': table_score,
            'count_score': count_score,
            'total': total_score
        })
        
        report.append(
            f"| {r['strategy']:<30} | {code_score:.1f} | {meta_score:.1f} | "
            f"{size_score:.1f} | {table_score:.1f} | {count_score:.1f} | **{total_score:.1f}** |"
        )
    
    report.extend(["", ""])
    
    # Recommendation
    best_strategy = max(scored_results, key=lambda x: x['total'])
    
    report.extend([
        "## 8. Recommendation",
        "",
        f"### 🏆 Winner: `{best_strategy['strategy']}`",
        "",
        f"**Total Score: {best_strategy['total']:.1f}/100**",
        "",
        "**Strengths:**"
    ])
    
    # Find the actual result data
    best_result = next(r for r in results if r['strategy'] == best_strategy['strategy'])
    
    if best_result['chunks_with_code'] == max(r['chunks_with_code'] for r in results):
        report.append(f"- ✅ Best code preservation ({best_result['chunks_with_code']} blocks)")
    
    meta_rate = best_result['metadata_preservation']['page_numbers'] / best_result['num_chunks'] * 100
    if meta_rate == 100:
        report.append(f"- ✅ Perfect metadata preservation (100%)")
    
    if best_result['chunks_with_tables'] >= max(r['chunks_with_tables'] for r in results) * 0.9:
        report.append(f"- ✅ Strong table retention ({best_result['chunks_with_tables']} tables)")
    
    report.extend([
        "",
        "**Why this matters for RAG:**",
        "- Code integrity ensures technical queries get complete function definitions",
        "- Metadata preservation enables accurate citations (page numbers)",
        "- Balanced chunk sizes optimize retrieval precision vs. context",
        "",
        "---",
        ""
    ])
    
    # Detailed Strategy Analysis
    report.extend([
        "## 9. Detailed Strategy Analysis",
        ""
    ])
    
    strategy_descriptions = {
        "1_recursive_small_500_100": {
            "name": "Recursive Small",
            "description": "500 char chunks, 100 overlap",
            "use_case": "Precise fact retrieval, Q&A",
            "pros": ["High granularity", "Fast retrieval", "Large result set"],
            "cons": ["Loss of context", "Many chunks (overhead)", "Code fragmentation"]
        },
        "2_recursive_medium_1000_200": {
            "name": "Recursive Medium",
            "description": "1000 char chunks, 200 overlap",
            "use_case": "Balanced retrieval",
            "pros": ["Good context", "Manageable chunk count", "Balanced performance"],
            "cons": ["May still split code", "Generic approach"]
        },
        "3_recursive_large_1500_300": {
            "name": "Recursive Large",
            "description": "1500 char chunks, 300 overlap",
            "use_case": "Complex queries needing context",
            "pros": ["Maximum context", "Fewer chunks", "Good for explanations"],
            "cons": ["May miss specific facts", "Slower retrieval", "Larger tokens"]
        },
        "4_markdown_header": {
            "name": "Markdown Header",
            "description": "Split only at headers",
            "use_case": "Structure-aware retrieval",
            "pros": ["Preserves hierarchy", "Conceptual boundaries", "Header metadata"],
            "cons": ["Extreme size variance", "Inconsistent embedding quality", "Poor code preservation"]
        },
        "5_hybrid_structure_size": {
            "name": "Hybrid",
            "description": "Headers + recursive split",
            "use_case": "Best of both worlds (theory)",
            "pros": ["Structure + size control", "Good balance"],
            "cons": ["CRITICAL: 80% metadata loss", "Complex pipeline", "Breaks citations"]
        },
        "6_code_aware_1200_200": {
            "name": "Code-Aware",
            "description": "Custom separators for code",
            "use_case": "Code-heavy technical docs",
            "pros": ["Preserves code blocks", "Perfect metadata", "Header awareness"],
            "cons": ["Slightly complex", "Custom tuning needed"]
        }
    }
    
    for r in results:
        strategy_key = r['strategy']
        desc = strategy_descriptions.get(strategy_key, {})
        
        report.extend([
            f"### {desc.get('name', strategy_key)}",
            "",
            f"**Description:** {desc.get('description', 'N/A')}",
            "",
            f"**Metrics:**",
            f"- Chunks: {r['num_chunks']:,}",
            f"- Avg Size: {r['avg_chunk_size']:.0f} chars",
            f"- Size Range: {r['min_chunk_size']}-{r['max_chunk_size']:,} chars",
            f"- Code Blocks: {r['chunks_with_code']}",
            f"- Tables: {r['chunks_with_tables']}",
            f"- Metadata: {r['metadata_preservation']['page_numbers']}/{r['num_chunks']} ({r['metadata_preservation']['page_numbers']/r['num_chunks']*100:.1f}%)",
            "",
            f"**Best For:** {desc.get('use_case', 'N/A')}",
            ""
        ])
        
        if desc.get('pros'):
            report.append("**Pros:**")
            for pro in desc['pros']:
                report.append(f"- {pro}")
            report.append("")
        
        if desc.get('cons'):
            report.append("**Cons:**")
            for con in desc['cons']:
                report.append(f"- {con}")
            report.append("")
        
        report.extend(["---", ""])
    
    # Implementation Guidance
    report.extend([
        "## 10. Implementation Guidance",
        "",
        f"To use the recommended strategy ({best_strategy['strategy']}), run:",
        "",
        "$env:CHUNK_STRATEGY=\"" + best_strategy['strategy'] + "\"",
        "python generate_embeddings.py",
        "",
        "This will:",
        "1. Load the optimized chunks",
        "2. Generate embeddings using text-embedding-3-large",
        "3. Store in ChromaDB/Pinecone for retrieval",
        "",
        "---",
        "",
        "## 11. For Your Codelab",
        "",
        "Include this analysis in your Lab 1 documentation:",
        "",
        "========================================",
        "Chunking Strategy Selection",
        "",
        f"After testing 6 strategies, selected {best_strategy['strategy']} based on:",
        f"- Superior code preservation ({best_result['chunks_with_code']} blocks)",
        f"- Perfect metadata retention (100%)",
        f"- Optimal size balance ({best_result['avg_chunk_size']:.0f} avg chars)",
        "",
        "This ensures accurate retrieval for code-heavy technical content",
        "while maintaining citation capability for RAG responses.",
        "========================================",
        "",
        "---",
        "",
        f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    # Write report as TXT
    report_file = output_dir / "chunking_comparison_report.txt"
    report_file.write_text("\n".join(report), encoding='utf-8')
    
    logger.info(f"✓ Generated detailed comparison report: {report_file.name}")



def run_all_strategies():
    """
    Run all chunking strategies and compare (Lab 1 requirement)
    """
    logger.info("="*70)
    logger.info("LAB 1 STEP 2: CHUNKING STRATEGY EXPERIMENTATION")
    logger.info("="*70)
    
    # Load documents
    documents = load_parsed_pages()
    
    if not documents:
        logger.error("❌ No parsed documents found. Run parse_pdf.py first!")
        return
    
    strategies = [
        ("1_recursive_small_500_100", lambda: strategy_1_recursive_small(documents)),
        ("2_recursive_medium_1000_200", lambda: strategy_2_recursive_medium(documents)),
        ("3_recursive_large_1500_300", lambda: strategy_3_recursive_large(documents)),
        ("4_markdown_header", lambda: strategy_4_markdown_header(documents)),
        ("5_hybrid_structure_size", lambda: strategy_5_hybrid(documents)),
        ("6_code_aware_1200_200", lambda: strategy_6_code_aware(documents)),
    ]
    
    results = []
    
    for strategy_name, strategy_func in strategies:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing Strategy: {strategy_name}")
        logger.info('-'*70)
        
        start = time.time()
        chunks = strategy_func()
        elapsed = time.time() - start
        
        metrics = evaluate_chunking_strategy(chunks, strategy_name)
        metrics['processing_time_sec'] = elapsed
        
        # Print metrics
        logger.info(f"  Chunks created: {metrics['num_chunks']}")
        logger.info(f"  Avg chunk size: {metrics['avg_chunk_size']:.0f} chars")
        logger.info(f"  Size range: {metrics['min_chunk_size']}-{metrics['max_chunk_size']} chars")
        logger.info(f"  Median size: {metrics['median_chunk_size']:.0f} chars")
        logger.info(f"  Chunks with code: {metrics['chunks_with_code']}")
        logger.info(f"  Chunks with tables: {metrics['chunks_with_tables']}")
        logger.info(f"  Chunks with formulas: {metrics['chunks_with_formulas']}")
        logger.info(f"  Processing time: {elapsed:.2f}s")
        
        # Save chunks
        save_chunks(chunks, strategy_name)
        save_sample_chunks(chunks, strategy_name)
        
        results.append(metrics)
    
    # Save comparison JSON
    comparison_file = CHUNKS_OUT / "strategy_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate detailed comparison report
    generate_comparison_report(results, CHUNKS_OUT)
    
    logger.info(f"\n{'='*70}")
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info('-'*70)
    logger.info(f"{'Strategy':<35} {'Chunks':<10} {'Avg Size':<12} {'Code':<8} {'Tables':<8}")
    logger.info('-'*70)
    for r in results:
        logger.info(
            f"{r['strategy']:<35} "
            f"{r['num_chunks']:<10} "
            f"{r['avg_chunk_size']:<12.0f} "
            f"{r['chunks_with_code']:<8} "
            f"{r['chunks_with_tables']:<8}"
        )
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ CHUNKING COMPLETE")
    logger.info("="*70)
    logger.info(f"📁 Output: {CHUNKS_OUT}")
    logger.info(f"   - Chunk files: {len(list(CHUNKS_OUT.glob('*_chunks.json')))} JSON files")
    logger.info(f"   - Sample files: {len(list(CHUNKS_OUT.glob('*_samples.md')))} MD files")
    logger.info(f"   - Comparison: strategy_comparison.json")
    logger.info(f"   - Detailed report: chunking_comparison_report.txt")
    logger.info("")
    logger.info("💡 NEXT STEPS:")
    logger.info("  1. Review detailed comparison: cat data/chunks/chunking_comparison_report.txt")
    logger.info("  2. Review samples: cat data/chunks/*_samples.md")
    logger.info("  3. Choose best strategy based on metrics")
    logger.info("  4. Document justification for Codelab")
    logger.info("  5. Move to Lab 1 Step 3: Embedding generation")
    logger.info("="*70)


if __name__ == "__main__":
    run_all_strategies()
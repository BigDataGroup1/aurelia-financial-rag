"""
Lab 1 Validation Script - AURELIA Project
Validates all outputs from Lab 1 before moving to Lab 3

UPDATED: Handles partial page processing and ChromaDB dimension issues
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Expected paths
RAW_PDF = PROJECT_ROOT / "data" / "raw_pdfs" / "fintbx.pdf"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PAGES_DIR = PROCESSED / "pages"
FIGURES_DIR = PROCESSED / "figures"
METADATA_DIR = PROCESSED / "metadata"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
CHROMADB_DIR = PROJECT_ROOT / "data" / "chromadb"

# Expected files
COMBINED_MD = PROCESSED / "fintbx_parsed.md"
MANIFEST_CSV = PROCESSED / "manifest.csv"
EXTRACTION_META = METADATA_DIR / "extraction_metadata.json"
STRATEGY_COMPARISON = CHUNKS_DIR / "strategy_comparison.json"
COMPARISON_REPORT = CHUNKS_DIR / "chunking_comparison_report.txt"

# Test results
test_results = []


class ValidationTest:
    """Container for test results"""
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed = False
        self.message = ""
        self.details = {}
    
    def mark_pass(self, message: str = "", **details):
        self.passed = True
        self.message = message
        self.details = details
    
    def mark_fail(self, message: str, **details):
        self.passed = False
        self.message = message
        self.details = details


def print_header(text: str):
    """Print formatted header"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{text}")
    logger.info(f"{'='*70}")


def print_section(text: str):
    logger.info(f"\n{'-'*70}")
    logger.info(f"{text}")
    logger.info(f"{'-'*70}")


def test_file_exists(name: str, path: Path, category: str = "Files") -> ValidationTest:
    """Test if a file exists"""
    test = ValidationTest(name, category)
    
    if path.exists():
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            test.mark_pass(f"Found ({size_mb:.2f} MB)", path=str(path))
        else:
            test.mark_pass(f"Found (directory)", path=str(path))
    else:
        test.mark_fail(f"Not found: {path}", path=str(path))
    
    return test


def test_directory_contents(name: str, path: Path, min_files: int, pattern: str = "*", category: str = "Files", note: str = "") -> ValidationTest:
    """Test if directory has minimum number of files"""
    test = ValidationTest(name, category)
    
    if not path.exists():
        test.mark_fail(f"Directory not found: {path}", path=str(path))
        return test
    
    files = list(path.glob(pattern))
    count = len(files)
    
    if count >= min_files:
        msg = f"Found {count} files (expected ≥{min_files})"
        if note:
            msg += f" - {note}"
        test.mark_pass(msg, count=count, path=str(path))
    else:
        test.mark_fail(f"Found only {count} files (expected ≥{min_files})", count=count, path=str(path))
    
    return test


def test_json_structure(name: str, path: Path, required_keys: List[str], category: str = "Data Integrity", is_array: bool = False) -> ValidationTest:
    """Test if JSON has required structure"""
    test = ValidationTest(name, category)
    
    if not path.exists():
        test.mark_fail(f"File not found: {path}", path=str(path))
        return test
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle array of objects
        if is_array:
            if not isinstance(data, list):
                test.mark_fail(f"Expected array, got {type(data).__name__}", path=str(path))
                return test
            
            if not data:
                test.mark_fail(f"Array is empty", path=str(path))
                return test
            
            # Check first item
            first_item = data[0]
            missing_keys = [key for key in required_keys if key not in first_item]
            
            if missing_keys:
                test.mark_fail(f"Missing keys in array items: {missing_keys}", path=str(path))
            else:
                test.mark_pass(f"{len(data)} items, all required keys present", count=len(data), path=str(path))
        else:
            # Handle single object
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                test.mark_fail(f"Missing keys: {missing_keys}", path=str(path))
            else:
                test.mark_pass(f"All required keys present", keys=list(data.keys()), path=str(path))
    
    except json.JSONDecodeError as e:
        test.mark_fail(f"Invalid JSON: {e}", path=str(path))
    except Exception as e:
        test.mark_fail(f"Error reading file: {e}", path=str(path))
    
    return test


def test_chunks_validity(name: str, strategy: str, category: str = "Chunks") -> ValidationTest:
    """Test if chunks are valid for a strategy"""
    test = ValidationTest(name, category)
    
    chunks_file = CHUNKS_DIR / f"{strategy}_chunks.json"
    
    if not chunks_file.exists():
        test.mark_fail(f"Chunks file not found: {chunks_file}", strategy=strategy)
        return test
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        if not chunks:
            test.mark_fail("Chunks list is empty", strategy=strategy)
            return test
        
        # Check first chunk structure
        first_chunk = chunks[0]
        required_keys = ['chunk_id', 'content', 'metadata', 'char_count']
        missing_keys = [key for key in required_keys if key not in first_chunk]
        
        if missing_keys:
            test.mark_fail(f"Missing keys in chunks: {missing_keys}", strategy=strategy)
            return test
        
        # Validate metadata preservation
        chunks_with_page = sum(1 for c in chunks if 'page' in c['metadata'])
        metadata_rate = chunks_with_page / len(chunks) * 100
        
        # Check for code preservation
        chunks_with_code = sum(1 for c in chunks if '```' in c['content'])
        
        test.mark_pass(
            f"{len(chunks)} chunks, {metadata_rate:.1f}% with page metadata",
            count=len(chunks),
            metadata_rate=metadata_rate,
            chunks_with_code=chunks_with_code,
            strategy=strategy
        )
    
    except Exception as e:
        test.mark_fail(f"Error validating chunks: {e}", strategy=strategy)
    
    return test


def test_embeddings_validity(name: str, strategy: str, category: str = "Embeddings") -> ValidationTest:
    """Test if embeddings are valid"""
    test = ValidationTest(name, category)
    
    embeddings_file = EMBEDDINGS_DIR / f"{strategy}_embeddings.json"
    
    if not embeddings_file.exists():
        test.mark_fail(f"Embeddings file not found", strategy=strategy, path=str(embeddings_file))
        return test
    
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embedded_chunks = json.load(f)
        
        if not embedded_chunks:
            test.mark_fail("Embeddings list is empty", strategy=strategy)
            return test
        
        # Check structure
        first_emb = embedded_chunks[0]
        required_keys = ['chunk_id', 'content', 'metadata', 'embedding', 'embedding_model', 'embedding_dimensions']
        missing_keys = [key for key in required_keys if key not in first_emb]
        
        if missing_keys:
            test.mark_fail(f"Missing keys: {missing_keys}", strategy=strategy)
            return test
        
        # Validate embedding dimensions
        expected_dim = 3072  # text-embedding-3-large
        actual_dim = len(first_emb['embedding'])
        
        if actual_dim != expected_dim:
            test.mark_fail(
                f"Wrong embedding dimensions: {actual_dim} (expected {expected_dim})",
                strategy=strategy
            )
            return test
        
        # Check all embeddings have same dimension
        all_dims = [len(e['embedding']) for e in embedded_chunks[:10]]  # Check first 10
        if len(set(all_dims)) > 1:
            test.mark_fail(f"Inconsistent dimensions: {set(all_dims)}", strategy=strategy)
            return test
        
        test.mark_pass(
            f"{len(embedded_chunks)} embeddings, {actual_dim}D vectors",
            count=len(embedded_chunks),
            dimensions=actual_dim,
            model=first_emb['embedding_model'],
            strategy=strategy
        )
    
    except Exception as e:
        test.mark_fail(f"Error validating embeddings: {e}", strategy=strategy)
    
    return test


def test_chromadb_collection(name: str, collection_name: str = "fintbx", category: str = "Vector Stores") -> ValidationTest:
    """Test ChromaDB collection"""
    test = ValidationTest(name, category)
    
    if not CHROMADB_DIR.exists():
        test.mark_fail(f"ChromaDB directory not found: {CHROMADB_DIR}")
        return test
    
    try:
        import chromadb
        
        client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        
        # Check if collection exists
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            test.mark_fail(f"Collection '{collection_name}' not found", collections=collection_names)
            return test
        
        # Get collection
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        
        if count == 0:
            test.mark_fail(f"Collection is empty", collection=collection_name)
            return test
        
        # Test a sample query
        try:
            # Try to query with pre-computed embeddings
            # If collection was created with embedding_function=None, this should work
            sample_embedding = [0.0] * 3072  # Dummy 3072D vector
            results = collection.query(
                query_embeddings=[sample_embedding],
                n_results=1
            )
            
            test.mark_pass(
                f"{count} vectors in ChromaDB, queries working",
                count=count,
                collection=collection_name
            )
        except Exception as query_error:
            # Check if it's a dimension mismatch
            if "dimension" in str(query_error).lower():
                test.mark_fail(
                    f"Dimension mismatch: {query_error}. "
                    "Collection created with wrong embedding function. "
                    "Run: python src/lab1/fix_chromadb.py",
                    collection=collection_name,
                    count=count
                )
            else:
                test.mark_fail(f"Query test failed: {query_error}", collection=collection_name)
    
    except ImportError:
        test.mark_fail("ChromaDB not installed (pip install chromadb)")
    except Exception as e:
        test.mark_fail(f"Error accessing ChromaDB: {e}")
    
    return test


def test_pinecone_index(name: str, index_name: str = "fintbx", category: str = "Vector Stores") -> ValidationTest:
    """Test Pinecone index"""
    test = ValidationTest(name, category)
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        test.mark_fail("PINECONE_API_KEY not set (optional)", optional=True)
        return test
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if index_name not in existing_indexes:
            test.mark_fail(f"Index '{index_name}' not found", indexes=existing_indexes)
            return test
        
        # Get index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            test.mark_fail(f"Index is empty", index=index_name)
            return test
        
        test.mark_pass(
            f"{total_vectors} vectors in Pinecone",
            count=total_vectors,
            index=index_name
        )
    
    except ImportError:
        test.mark_fail("Pinecone not installed (pip install pinecone-client)", optional=True)
    except Exception as e:
        test.mark_fail(f"Error accessing Pinecone: {e}")
    
    return test


def run_all_validations():
    """Run all validation tests"""
    
    print_header("LAB 1 VALIDATION - AURELIA PROJECT")
    
    # Check processing scope
    logger.info("\n📋 PROCESSING SCOPE:")
    if EXTRACTION_META.exists():
        with open(EXTRACTION_META, 'r') as f:
            meta = json.load(f)
            total_pages = meta.get('total_pages', 'Unknown')
            logger.info(f"   Pages processed: {total_pages}")
            logger.info(f"   ℹ️  Partial processing is acceptable for development/testing")
    
    # Category 1: File Structure
    print_section("1. FILE STRUCTURE & EXISTENCE")
    
    tests = [
        test_file_exists("Raw PDF", RAW_PDF, "Files"),
        test_file_exists("Combined Markdown", COMBINED_MD, "Files"),
        test_file_exists("Manifest CSV", MANIFEST_CSV, "Files"),
        test_file_exists("Extraction Metadata", EXTRACTION_META, "Files"),
        test_file_exists("Strategy Comparison", STRATEGY_COMPARISON, "Files"),
        test_file_exists("Comparison Report", COMPARISON_REPORT, "Files"),
    ]
    test_results.extend(tests)
    
    for t in tests:
        status = "✅" if t.passed else "❌"
        logger.info(f"{status} {t.name}: {t.message}")
    
    # Category 2: Directories with Content
    print_section("2. DIRECTORY CONTENTS")
    
    tests = [
        test_directory_contents("Page Markdowns", PAGES_DIR, 10, "page_*.md", "Files", "Partial processing OK"),
        test_directory_contents("Figures Extracted", FIGURES_DIR, 1, "*.png", "Files", "Depends on page range"),
        test_directory_contents("Chunk Files", CHUNKS_DIR, 6, "*_chunks.json", "Files"),
        test_directory_contents("Sample Files", CHUNKS_DIR, 6, "*_samples.md", "Files"),
    ]
    test_results.extend(tests)
    
    for t in tests:
        status = "✅" if t.passed else "❌"
        logger.info(f"{status} {t.name}: {t.message}")
    
    # Category 3: Data Integrity
    print_section("3. DATA INTEGRITY - METADATA")
    
    tests = [
        test_json_structure(
            "Extraction Metadata",
            EXTRACTION_META,
            ['total_pages', 'total_images', 'total_tables', 'pages'],
            "Data Integrity"
        ),
        test_json_structure(
            "Strategy Comparison",
            STRATEGY_COMPARISON,
            ['strategy', 'num_chunks', 'avg_chunk_size'],
            "Data Integrity",
            is_array=True  # ← FIXED: This is an array of strategy results
        ),
    ]
    test_results.extend(tests)
    
    for t in tests:
        status = "✅" if t.passed else "❌"
        logger.info(f"{status} {t.name}: {t.message}")
    
    # Category 4: Chunks Validation
    print_section("4. CHUNKS VALIDATION")
    
    strategies = [
        "1_recursive_small_500_100",
        "2_recursive_medium_1000_200",
        "3_recursive_large_1500_300",
        "4_markdown_header",
        "5_hybrid_structure_size",
        "6_code_aware_1200_200",
    ]
    
    for strategy in strategies:
        test = test_chunks_validity(f"Strategy: {strategy}", strategy, "Chunks")
        test_results.append(test)
        status = "✅" if test.passed else "❌"
        logger.info(f"{status} {test.name}: {test.message}")
    
    # Category 5: Embeddings Validation
    print_section("5. EMBEDDINGS VALIDATION")
    
    # Check which strategy has embeddings (should be at least one)
    embedding_files = list(EMBEDDINGS_DIR.glob("*_embeddings.json"))
    
    if not embedding_files:
        test = ValidationTest("Embeddings Generated", "Embeddings")
        test.mark_fail("No embedding files found. Run generate_embeddings.py")
        test_results.append(test)
        logger.info(f"❌ {test.name}: {test.message}")
    else:
        for emb_file in embedding_files:
            strategy = emb_file.stem.replace('_embeddings', '')
            test = test_embeddings_validity(f"Strategy: {strategy}", strategy, "Embeddings")
            test_results.append(test)
            status = "✅" if test.passed else "❌"
            logger.info(f"{status} {test.name}: {test.message}")
    
    # Category 6: Vector Stores
    print_section("6. VECTOR STORES")
    
    tests = [
        test_chromadb_collection("ChromaDB Collection", "fintbx", "Vector Stores"),
        test_pinecone_index("Pinecone Index", "fintbx", "Vector Stores"),
    ]
    test_results.extend(tests)
    
    for t in tests:
        status = "✅" if t.passed else "❌"
        optional = " (optional)" if t.details.get('optional') else ""
        logger.info(f"{status} {t.name}{optional}: {t.message}")
    
    # Final Summary
    print_summary()


def print_summary():
    """Print final validation summary"""
    
    print_header("VALIDATION SUMMARY")
    
    # Count by category
    categories = {}
    for test in test_results:
        if test.category not in categories:
            categories[test.category] = {'passed': 0, 'failed': 0, 'total': 0}
        
        categories[test.category]['total'] += 1
        if test.passed:
            categories[test.category]['passed'] += 1
        else:
            categories[test.category]['failed'] += 1
    
    # Print category breakdown
    logger.info("\nResults by Category:")
    logger.info("-" * 70)
    
    total_passed = 0
    total_failed = 0
    
    for category, stats in categories.items():
        passed = stats['passed']
        total = stats['total']
        status = "✅" if stats['failed'] == 0 else "⚠️" if passed > 0 else "❌"
        
        logger.info(f"{status} {category:<30} {passed}/{total} passed")
        
        total_passed += passed
        total_failed += stats['failed']
    
    # Overall status
    logger.info("\n" + "=" * 70)
    
    total_tests = total_passed + total_failed
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    logger.info(f"OVERALL: {total_passed}/{total_tests} tests passed ({pass_rate:.1f}%)")
    logger.info("=" * 70)
    
    # Critical failures
    critical_failures = [
        t for t in test_results 
        if not t.passed and not t.details.get('optional')
    ]
    
    if critical_failures:
        logger.info("\n⚠️  CRITICAL FAILURES:")
        for t in critical_failures:
            logger.info(f"   ❌ {t.category} - {t.name}: {t.message}")
        
        logger.info("\n" + "=" * 70)
        logger.info("❌ LAB 1 VALIDATION FAILED")
        logger.info("=" * 70)
        logger.info("\n🔧 REQUIRED ACTIONS:")
        logger.info("   1. Review failed tests above")
        logger.info("   2. If ChromaDB dimension mismatch, run:")
        logger.info("      python src/lab1/fix_chromadb.py")
        logger.info("   3. Ensure all Lab 1 scripts have been run:")
        logger.info("      - python src/lab1/parse_pdf_final.py")
        logger.info("      - python src/lab1/chunk_documents.py")
        logger.info("      - python src/lab1/generate_embeddings.py")
        logger.info("   4. Re-run this validation script")
        logger.info("=" * 70)
        
        return False
    else:
        logger.info("\n" + "=" * 70)
        logger.info("✅ LAB 1 VALIDATION PASSED!")
        logger.info("=" * 70)
        logger.info("\n🎉 All critical components are ready!")
        logger.info("\n📊 Lab 1 Summary:")
        
        # Extract some stats
        if EXTRACTION_META.exists():
            with open(EXTRACTION_META, 'r') as f:
                meta = json.load(f)
                logger.info(f"   - Pages processed: {meta.get('total_pages', 'N/A')}")
                logger.info(f"   - Images extracted: {meta.get('total_images', 'N/A')}")
                logger.info(f"   - Tables extracted: {meta.get('total_tables', 'N/A')}")
                logger.info(f"   - Code blocks: {meta.get('total_code_blocks', 'N/A')}")
        
        # Chunk stats
        passed_chunk_tests = [t for t in test_results if t.category == "Chunks" and t.passed]
        if passed_chunk_tests:
            logger.info(f"   - Chunking strategies: {len(passed_chunk_tests)}")
        
        # Embedding stats
        passed_emb_tests = [t for t in test_results if t.category == "Embeddings" and t.passed]
        if passed_emb_tests:
            logger.info(f"   - Embedding strategies: {len(passed_emb_tests)}")
            # Get total vectors
            first_emb_test = passed_emb_tests[0]
            if 'count' in first_emb_test.details:
                logger.info(f"   - Vector count: {first_emb_test.details['count']}")
        
        logger.info("\n🚀 READY FOR LAB 3!")
        logger.info("   You can now proceed to build the FastAPI RAG service")
        logger.info("=" * 70)
        
        return True


def main():
    """Main validation function"""
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️  OPENAI_API_KEY not set (needed for embeddings)")
    
    # Run validations
    success = run_all_validations()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
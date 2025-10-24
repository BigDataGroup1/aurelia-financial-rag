# """
# Lab 2: fintbx_ingest_dag - Orchestrates Your Python Modules
# Imports and runs: parse_pdf.py, chunk_documents.py, generate_embeddings.py
# Schedule: Weekly refresh
# """
# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
# from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
# from airflow.providers.google.cloud.operators.gcs import GCSFileTransformOperator
# from airflow.models import Variable
# import logging

# # Configuration
# GCS_BUCKET = Variable.get("gcs_bucket", default_var="aurelia-rag-data")
# PROJECT_ID = Variable.get("gcp_project_id", default_var="aurelia-financial-rag-475403")

# default_args = {
#     'owner': 'aurelia-team',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
#     'execution_timeout': timedelta(hours=8),
# }


# def parse_pdf_wrapper(**context):
#     """
#     Wrapper task that calls YOUR parse_pdf.py module
#     Downloads PDF directly from GCS to avoid worker isolation issues
#     """
#     import os
#     import sys
#     from pathlib import Path
#     import json
#     from google.cloud import storage
    
#     logging.info("="*70)
#     logging.info("RUNNING YOUR PARSE_PDF.PY MODULE")
#     logging.info("="*70)
    
#     # Download PDF directly in this task (avoids worker isolation)
#     bucket_name = Variable.get("gcs_bucket")
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob('raw_pdfs/fintbx.pdf')
    
#     pdf_download_path = Path("/tmp/fintbx_downloaded.pdf")
#     blob.download_to_filename(str(pdf_download_path))
#     logging.info(f"✓ Downloaded PDF from GCS: {pdf_download_path}")
    
#     # Import YOUR parsing module
#     import  parse_pdf
    
#     # Set up environment for your code
#     os.environ['PARSE_MAX_PAGES'] = Variable.get("parse_page_limit", default_var="20")
    
#     # Create temp directories matching your code's structure
#     temp_root = Path("/tmp/aurelia_project")
#     temp_root.mkdir(exist_ok=True)
    
#     raw_pdfs_dir = temp_root / "data" / "raw_pdfs"
#     processed_dir = temp_root / "data" / "processed"
#     pages_dir = processed_dir / "pages"
#     figures_dir = processed_dir / "figures"
    
#     raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
#     pages_dir.mkdir(parents=True, exist_ok=True)
#     figures_dir.mkdir(parents=True, exist_ok=True)
    
#     # Copy downloaded PDF to location your code expects
#     import shutil
#     shutil.copy(str(pdf_download_path), raw_pdfs_dir / "fintbx.pdf")
#     logging.info(f"✓ PDF copied to: {raw_pdfs_dir / 'fintbx.pdf'}")
    
#     # Modify parse_pdf module paths to use temp location
#     parse_pdf.PROJECT_ROOT = temp_root
#     parse_pdf.RAW_PDF = raw_pdfs_dir / "fintbx.pdf"
#     parse_pdf.OUT_PROCESSED = processed_dir
#     parse_pdf.OUT_PAGES = pages_dir
#     parse_pdf.OUT_FIGS = figures_dir
#     parse_pdf.OUT_META = processed_dir / "metadata"
#     parse_pdf.OUT_META.mkdir(exist_ok=True)
    
#     # IMPORTANT: Update module-level variables that depend on OUT_PROCESSED
#     parse_pdf.COMBINED_MD = processed_dir / "fintbx_parsed.md"
#     parse_pdf.MANIFEST_CSV = processed_dir / "manifest.csv"
    
#     # Run YOUR main() function
#     logging.info("Calling your parse_pdf.main() function...")
#     logging.info(f"Processing up to {os.environ.get('PARSE_MAX_PAGES', 'ALL')} pages...")

#     parse_pdf.main()
    
#     # Your code creates the complete output structure
#     num_pages = len(list(pages_dir.glob("page_*.md")))
#     logging.info("✓ Your parse_pdf.py completed : {num_pages} pages processed")
    
#     # Upload outputs immediately (same worker, same task!)
#     from google.cloud import storage
    
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(Variable.get("gcs_bucket"))
    
#     # Get execution date for folder naming
#     ds = context['ds']
    
#     logging.info(f"Uploading outputs to GCS bucket: {Variable.get('gcs_bucket')}")
    
#     # Upload all page markdown files
#     for page_file in sorted(pages_dir.glob("page_*.md")):
#         blob_path = f"processed/{ds}/pages/{page_file.name}"
#         blob = bucket.blob(blob_path)
#         blob.upload_from_filename(str(page_file))
#     logging.info(f"✓ Uploaded {len(list(pages_dir.glob('page_*.md')))} page files")
    
#     # Upload all figures
#     for fig_file in figures_dir.glob("*"):
#         blob_path = f"processed/{ds}/figures/{fig_file.name}"
#         blob = bucket.blob(blob_path)
#         blob.upload_from_filename(str(fig_file))
#     logging.info(f"✓ Uploaded {len(list(figures_dir.glob('*')))} figures")
    
#     # Upload combined markdown
#     combined_md = processed_dir / "fintbx_parsed.md"
#     if combined_md.exists():
#         blob = bucket.blob(f"processed/{ds}/fintbx_parsed.md")
#         blob.upload_from_filename(str(combined_md))
#         logging.info("✓ Uploaded fintbx_parsed.md")
    
#     # Upload manifest
#     manifest = processed_dir / "manifest.csv"
#     if manifest.exists():
#         blob = bucket.blob(f"processed/{ds}/manifest.csv")
#         blob.upload_from_filename(str(manifest))
#         logging.info("✓ Uploaded manifest.csv")
    
#     # Upload extraction metadata
#     meta_file = processed_dir / "metadata" / "extraction_metadata.json"
#     if meta_file.exists():
#         blob = bucket.blob(f"processed/{ds}/metadata/extraction_metadata.json")
#         blob.upload_from_filename(str(meta_file))
#         logging.info("✓ Uploaded extraction_metadata.json")
    
#     num_pages = len(list(pages_dir.glob("page_*.md")))
    
#     context['ti'].xcom_push(key='num_pages', value=num_pages)
#     context['ti'].xcom_push(key='processed_dir', value=str(processed_dir))
    
#     return "parse_complete"


# def chunk_documents_wrapper(**context):
#     """
#     Wrapper task that calls YOUR chunk_documents.py module
#     """
#     from pathlib import Path
#     import json
    
#     logging.info("="*70)
#     logging.info("RUNNING YOUR CHUNK_DOCUMENTS.PY MODULE")
#     logging.info("="*70)
    
#     # Import YOUR chunking module
#     import chunk_documents
    
#     # Set up temp directories
#     temp_root = Path("/tmp/aurelia_project")
#     processed_dir = temp_root / "data" / "processed"
#     pages_dir = processed_dir / "pages"
#     chunks_dir = temp_root / "data" / "chunks"
#     chunks_dir.mkdir(parents=True, exist_ok=True)
    
#     # Update module paths
#     chunk_documents.PROJECT_ROOT = temp_root
#     chunk_documents.PROCESSED = processed_dir
#     chunk_documents.PAGES_DIR = pages_dir
#     chunk_documents.CHUNKS_OUT = chunks_dir
    
#     # Load pages (created by parse_pdf task)
#     logging.info("Loading parsed pages...")
#     documents = chunk_documents.load_parsed_pages()
    
#     # Run YOUR strategy_6_code_aware function
#     logging.info("Running your code-aware chunking strategy...")
#     chunks = chunk_documents.strategy_6_code_aware(documents)
    
#     # Evaluate with YOUR metrics function
#     metrics = chunk_documents.evaluate_chunking_strategy(chunks, "6_code_aware_1200_200")
    
#     # Save using YOUR save function
#     chunk_documents.save_chunks(chunks, "6_code_aware_1200_200")
    
#     logging.info(f"✓ Your chunk_documents.py created {metrics['num_chunks']} chunks")
#     logging.info(f"  Avg size: {metrics['avg_chunk_size']:.0f} chars")
#     logging.info(f"  Code blocks: {metrics['chunks_with_code']}")
    
#     # Copy to /tmp for upload operators
#     import shutil
#     tmp_chunks_dir = Path("/tmp/chunks")
#     tmp_chunks_dir.mkdir(parents=True, exist_ok=True)  # Create directory first!
    
#     shutil.copy(
#         chunks_dir / "6_code_aware_1200_200_chunks.json",
#         tmp_chunks_dir / "chunks.json"
#     )
    
#     context['ti'].xcom_push(key='num_chunks', value=metrics['num_chunks'])
#     context['ti'].xcom_push(key='metrics', value=metrics)
    
#     return "chunk_complete"


# def generate_embeddings_wrapper(**context):
#     """
#     Wrapper task that calls YOUR generate_embeddings.py module
#     """
#     import os
#     from pathlib import Path
#     import json
    
#     logging.info("="*70)
#     logging.info("RUNNING YOUR GENERATE_EMBEDDINGS.PY MODULE")
#     logging.info("="*70)
    
#     # Set environment variables for your code
#     os.environ['OPENAI_API_KEY'] = Variable.get("openai_api_key")
#     os.environ['CHUNK_STRATEGY'] = "6_code_aware_1200_200"
    
#     # Import YOUR embedding module
#     import generate_embeddings
    
#     # Set up temp directories
#     temp_root = Path("/tmp/aurelia_project")
#     chunks_dir = temp_root / "data" / "chunks"
#     embeddings_dir = temp_root / "data" / "embeddings"
#     embeddings_dir.mkdir(parents=True, exist_ok=True)
    
#     # Update module paths
#     generate_embeddings.PROJECT_ROOT = temp_root
#     generate_embeddings.CHUNKS_DIR = chunks_dir
#     generate_embeddings.EMBEDDINGS_DIR = embeddings_dir
    
#     # Load chunks
#     logging.info("Loading chunks...")
#     chunks = generate_embeddings.load_chunks("6_code_aware_1200_200")
    
#     # Run YOUR embed_chunks function
#     logging.info("Running your embedding generation...")
#     embedded_chunks, metadata = generate_embeddings.embed_chunks(chunks, "6_code_aware_1200_200")
    
#     logging.info(f"✓ Your generate_embeddings.py created {len(embedded_chunks)} embeddings")
#     logging.info(f"  Model: {metadata['model']}")
#     logging.info(f"  Dimensions: {metadata['dimensions']}")
#     logging.info(f"  Cost: ${metadata['estimated_cost_usd']:.4f}")
    
#     # Copy to /tmp for upload operators
#     import shutil
#     tmp_embeddings_dir = Path("/tmp/embeddings")
#     tmp_embeddings_dir.mkdir(parents=True, exist_ok=True)  # Create directory first!
    
#     shutil.copy(
#         embeddings_dir / "6_code_aware_1200_200_embeddings.json",
#         tmp_embeddings_dir / "embeddings.json"
#     )
#     shutil.copy(
#         embeddings_dir / "6_code_aware_1200_200_embedding_metadata.json",
#         tmp_embeddings_dir / "embedding_metadata.json"
#     )
    
#     context['ti'].xcom_push(key='num_embeddings', value=len(embedded_chunks))
#     context['ti'].xcom_push(key='metadata', value=metadata)
    
#     return "embeddings_complete"


# # Define the DAG
# with DAG(
#     dag_id='fintbx_ingest_pipeline',
#     default_args=default_args,
#     description='Orchestrates YOUR Python modules: parse_pdf, chunk_documents, generate_embeddings',
#     schedule_interval='0 0 * * 0',
#     start_date=datetime(2025, 10, 1),
#     catchup=False,
#     tags=['production', 'weekly', 'ingestion', 'lab2'],
#     doc_md="""
#     # Financial Toolbox Ingestion Pipeline
    
#     This DAG orchestrates YOUR Lab 1 Python modules:
#     - parse_pdf.py (complete parsing logic)
#     - chunk_documents.py (strategy 6 code-aware)
#     - generate_embeddings.py (text-embedding-3-large)
    
#     Your exact local code runs in the cloud!
#     """
# ) as dag:
    
#     # Task 1: Parse PDF (downloads from GCS internally)
#     parse_pdf = PythonOperator(
#     task_id='run_your_parse_pdf',
#     python_callable=parse_pdf_wrapper,
#     provide_context=True,
#     execution_timeout=timedelta(hours=6),  # ← 2 hours for parsing 4000 pages
#     pool='default_pool',
# )
    
#     chunk_docs = PythonOperator(
#     task_id='run_your_chunk_documents',
#     python_callable=chunk_documents_wrapper,
#     provide_context=True,
#     execution_timeout=timedelta(hours=6),  # ← 1 hour for chunking
# )
    
#     embed_chunks = PythonOperator(
#     task_id='run_your_generate_embeddings',
#     python_callable=generate_embeddings_wrapper,
#     provide_context=True,
#     execution_timeout=timedelta(hours=6),  # ← 3 hours for embedding (SLOWEST - OpenAI API)
# )
#     # Upload complete output structure to GCS
#     upload_pages_folder = LocalFilesystemToGCSOperator(
#         task_id='upload_pages_folder',
#         src='/tmp/aurelia_project/data/processed/pages/*',
#         dst='processed/{{ ds }}/pages/',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_figures = LocalFilesystemToGCSOperator(
#         task_id='upload_figures',
#         src='/tmp/aurelia_project/data/processed/figures/*',
#         dst='processed/{{ ds }}/figures/',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_combined_md = LocalFilesystemToGCSOperator(
#         task_id='upload_combined_markdown',
#         src='/tmp/aurelia_project/data/processed/fintbx_parsed.md',
#         dst='processed/{{ ds }}/fintbx_parsed.md',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_manifest = LocalFilesystemToGCSOperator(
#         task_id='upload_manifest',
#         src='/tmp/aurelia_project/data/processed/manifest.csv',
#         dst='processed/{{ ds }}/manifest.csv',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_extraction_meta = LocalFilesystemToGCSOperator(
#         task_id='upload_extraction_metadata',
#         src='/tmp/aurelia_project/data/processed/metadata/extraction_metadata.json',
#         dst='processed/{{ ds }}/metadata/extraction_metadata.json',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_chunks = LocalFilesystemToGCSOperator(
#         task_id='upload_chunks',
#         src='/tmp/chunks/chunks.json',
#         dst='chunks/{{ ds }}/6_code_aware_1200_200_chunks.json',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_embeddings = LocalFilesystemToGCSOperator(
#         task_id='upload_embeddings',
#         src='/tmp/embeddings/embeddings.json',
#         dst='embeddings/{{ ds }}/embeddings.json',
#         bucket=GCS_BUCKET,
#     )
    
#     upload_embed_metadata = LocalFilesystemToGCSOperator(
#         task_id='upload_embedding_metadata',
#         src='/tmp/embeddings/embedding_metadata.json',
#         dst='embeddings/{{ ds }}/embedding_metadata.json',
#         bucket=GCS_BUCKET,
#     )
    
#     # Task flow
#     parse_pdf >> chunk_docs >> embed_chunks
    
#     # Upload parse_pdf outputs (exact local structure)
#     parse_pdf >> [upload_pages_folder, upload_figures, upload_combined_md, upload_manifest, upload_extraction_meta]
    
#     # Upload chunking outputs
#     chunk_docs >> upload_chunks
    
#     # Upload embedding outputs
#     embed_chunks >> [upload_embeddings, upload_embed_metadata]


"""
Lab 2: fintbx_ingest_dag - Orchestrates Your Python Modules
Imports and runs: parse_pdf.py, chunk_documents.py, generate_embeddings.py
Schedule: Weekly refresh
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.models import Variable
import logging


# Configuration
GCS_BUCKET = Variable.get("gcs_bucket", default_var="aurelia-rag-data")
PROJECT_ID = Variable.get("gcp_project_id", default_var="aurelia-financial-rag-475403")

default_args = {
    'owner': 'aurelia-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=8),
}


def parse_pdf_wrapper(**context):
    """
    Wrapper task that calls YOUR parse_pdf.py module and uploads outputs
    """
    import os
    from pathlib import Path
    from google.cloud import storage
    import parse_pdf
    import shutil

    logging.info("=" * 70)
    logging.info("RUNNING YOUR PARSE_PDF.PY MODULE")
    logging.info("=" * 70)

    bucket_name = Variable.get("gcs_bucket")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Download PDF directly
    pdf_path = Path("/tmp/fintbx_downloaded.pdf")
    blob = bucket.blob("raw_pdfs/fintbx.pdf")
    blob.download_to_filename(str(pdf_path))
    logging.info(f"✓ Downloaded PDF: {pdf_path}")

    # Temporary structure
    temp_root = Path("/tmp/aurelia_project")
    raw_dir = temp_root / "data" / "raw_pdfs"
    proc_dir = temp_root / "data" / "processed"
    pages_dir = proc_dir / "pages"
    figs_dir = proc_dir / "figures"
    meta_dir = proc_dir / "metadata"
    for d in [raw_dir, pages_dir, figs_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(pdf_path, raw_dir / "fintbx.pdf")
    parse_pdf.PROJECT_ROOT = temp_root
    parse_pdf.RAW_PDF = raw_dir / "fintbx.pdf"
    parse_pdf.OUT_PROCESSED = proc_dir
    parse_pdf.OUT_PAGES = pages_dir
    parse_pdf.OUT_FIGS = figs_dir
    parse_pdf.OUT_META = meta_dir
    parse_pdf.COMBINED_MD = proc_dir / "fintbx_parsed.md"
    parse_pdf.MANIFEST_CSV = proc_dir / "manifest.csv"

    parse_pdf.main()
    logging.info("✓ parse_pdf.main() completed")

    # Upload outputs immediately (before chunking)
    ds = context["ds"]
    def upload_dir(local_dir, gcs_prefix):
        for f in Path(local_dir).glob("*"):
            blob = bucket.blob(f"{gcs_prefix}/{f.name}")
            blob.upload_from_filename(str(f))
        logging.info(f"✓ Uploaded {len(list(Path(local_dir).glob('*')))} -> gs://{bucket_name}/{gcs_prefix}")

    upload_dir(pages_dir, f"processed/{ds}/pages")
    upload_dir(figs_dir, f"processed/{ds}/figures")
    upload_dir(meta_dir, f"processed/{ds}/metadata")

    if (proc_dir / "fintbx_parsed.md").exists():
        blob = bucket.blob(f"processed/{ds}/fintbx_parsed.md")
        blob.upload_from_filename(str(proc_dir / "fintbx_parsed.md"))
        logging.info("✓ Uploaded combined markdown")

    if (proc_dir / "manifest.csv").exists():
        blob = bucket.blob(f"processed/{ds}/manifest.csv")
        blob.upload_from_filename(str(proc_dir / "manifest.csv"))
        logging.info("✓ Uploaded manifest")

    logging.info("✅ parse_pdf_wrapper done and outputs uploaded.")
    return "parse_complete"


def chunk_documents_wrapper(**context):
    """
    Wrapper task that downloads pages from GCS, chunks them, and uploads output
    """
    import os
    from pathlib import Path
    from google.cloud import storage
    import chunk_documents
    import shutil

    ds = context["ds"]
    logging.info("=" * 70)
    logging.info("RUNNING YOUR CHUNK_DOCUMENTS.PY MODULE")
    logging.info("=" * 70)

    temp_root = Path("/tmp/aurelia_project")
    proc_dir = temp_root / "data" / "processed"
    pages_dir = proc_dir / "pages"
    chunks_dir = temp_root / "data" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Download pages from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    for blob in bucket.list_blobs(prefix=f"processed/{ds}/pages/"):
        if blob.name.endswith("/"):
            continue
        local_path = pages_dir / Path(blob.name).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
    logging.info(f"✓ Downloaded pages to {pages_dir}")

    # Point module paths
    chunk_documents.PROJECT_ROOT = temp_root
    chunk_documents.PROCESSED = proc_dir
    chunk_documents.PAGES_DIR = pages_dir
    chunk_documents.CHUNKS_OUT = chunks_dir

    chunk_documents.main()
    logging.info("✓ chunk_documents.main() completed")

    # Upload chunks to GCS
    for f in chunks_dir.glob("*.json"):
        blob = bucket.blob(f"chunks/{ds}/{f.name}")
        blob.upload_from_filename(str(f))
    logging.info(f"✓ Uploaded {len(list(chunks_dir.glob('*.json')))} chunk files")
    return "chunk_complete"


def generate_embeddings_wrapper(**context):
    """
    Wrapper task that downloads chunks from GCS, generates embeddings, and uploads them
    """
    import os
    from pathlib import Path
    from google.cloud import storage
    import generate_embeddings
    import shutil

    ds = context["ds"]
    logging.info("=" * 70)
    logging.info("RUNNING YOUR GENERATE_EMBEDDINGS.PY MODULE")
    logging.info("=" * 70)

    os.environ["OPENAI_API_KEY"] = Variable.get("openai_api_key")
    os.environ["CHUNK_STRATEGY"] = "6_code_aware_1200_200"

    temp_root = Path("/tmp/aurelia_project")
    chunks_dir = temp_root / "data" / "chunks"
    embeds_dir = temp_root / "data" / "embeddings"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeds_dir.mkdir(parents=True, exist_ok=True)

    # Download chunks from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    for blob in bucket.list_blobs(prefix=f"chunks/{ds}/"):
        if blob.name.endswith("/"):
            continue
        local_path = chunks_dir / Path(blob.name).name
        blob.download_to_filename(str(local_path))
    logging.info(f"✓ Downloaded chunks to {chunks_dir}")

    # Run embedding generation
    generate_embeddings.PROJECT_ROOT = temp_root
    generate_embeddings.CHUNKS_DIR = chunks_dir
    generate_embeddings.EMBEDDINGS_DIR = embeds_dir

    generate_embeddings.main()
    logging.info("✓ generate_embeddings.main() completed")

    # Upload embeddings back to GCS
    for f in embeds_dir.glob("*.json"):
        blob = bucket.blob(f"embeddings/{ds}/{f.name}")
        blob.upload_from_filename(str(f))
    logging.info(f"✓ Uploaded {len(list(embeds_dir.glob('*.json')))} embeddings files")
    return "embeddings_complete"


# ---------------------------
# DAG Definition
# ---------------------------
with DAG(
    dag_id="fintbx_ingest_pipeline",
    default_args=default_args,
    description="Orchestrates YOUR Python modules: parse_pdf, chunk_documents, generate_embeddings",
    schedule_interval="0 0 * * 0",
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=["production", "weekly", "ingestion", "lab2"],
) as dag:

    parse_pdf = PythonOperator(
        task_id="run_your_parse_pdf",
        python_callable=parse_pdf_wrapper,
        provide_context=True,
        execution_timeout=timedelta(hours=6),
    )

    chunk_docs = PythonOperator(
        task_id="run_your_chunk_documents",
        python_callable=chunk_documents_wrapper,
        provide_context=True,
        execution_timeout=timedelta(hours=3),
    )

    embed_chunks = PythonOperator(
        task_id="run_your_generate_embeddings",
        python_callable=generate_embeddings_wrapper,
        provide_context=True,
        execution_timeout=timedelta(hours=4),
    )

    # Dependencies
    parse_pdf >> chunk_docs >> embed_chunks

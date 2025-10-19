"""
Lab 2: fintbx_ingest_dag - Main Ingestion Pipeline
Orchestrates: PDF parsing → Chunking → Embedding → Storage
Schedule: Weekly refresh
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.models import Variable
import logging

# Configuration from Airflow Variables (set these in Airflow UI)
GCS_BUCKET = Variable.get("gcs_bucket", default_var="your-bucket-name")
PROJECT_ID = Variable.get("gcp_project_id", default_var="your-project-id")

default_args = {
    'owner': 'aurelia-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def parse_pdf_task(**context):
    """
    Parse PDF using pdfplumber + PyMuPDF
    Adapted from your parse_pdf_final.py
    """
    import os
    import re
    from pathlib import Path
    import json
    
    logging.info("="*70)
    logging.info("TASK 1: PDF PARSING")
    logging.info("="*70)
    
    # Paths in Airflow worker
    pdf_path = Path("/tmp/fintbx.pdf")
    output_dir = Path("/tmp/processed")
    figures_dir = output_dir / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import pdfplumber
        import fitz  # PyMuPDF
    except ImportError as e:
        logging.error(f"Missing library: {e}")
        logging.error("Ensure pdfplumber and pymupdf are in PyPI packages")
        raise
    
    # Helper functions (from your code)
    def detect_code_blocks(text):
        code_blocks = []
        lines = text.split('\n')
        in_code = False
        code_start, code_lines = 0, []
        
        for i, line in enumerate(lines):
            is_code = (
                line.strip().startswith('>>') or
                line.strip().startswith('%') or
                line.strip().startswith('function ') or
                re.match(r'^\s*(for|if|while|switch)\s+', line) or
                (line.strip() == 'end' and in_code) or
                (in_code and re.match(r'^\s+', line) and len(line.strip()) > 0)
            )
            
            if is_code and not in_code:
                in_code, code_start, code_lines = True, i, [line]
            elif is_code and in_code:
                code_lines.append(line)
            elif not is_code and in_code:
                in_code = False
                code_blocks.append((code_start, i - 1, '\n'.join(code_lines)))
                code_lines = []
        
        if in_code and code_lines:
            code_blocks.append((code_start, len(lines) - 1, '\n'.join(code_lines)))
        
        return code_blocks
    
    # Extract from PDF
    pymupdf_doc = fitz.open(str(pdf_path))
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        # Option 1: Check DAG run config (per-run override)
        dag_run_conf = context.get('dag_run').conf or {}
        page_limit_conf = dag_run_conf.get('page_limit')
        
        # Option 2: Check Airflow Variable (global setting)
        page_limit_var = Variable.get("parse_page_limit", default_var="20")
        
        # Use DAG config if provided, otherwise use variable
        if page_limit_conf:
            page_limit = min(int(page_limit_conf), total_pages)
            logging.info(f"Using DAG config page limit: {page_limit}")
        else:
            page_limit = min(int(page_limit_var), total_pages)
            logging.info(f"Using Variable page limit: {page_limit}")
        
        logging.info(f"Processing {page_limit} pages (out of {total_pages} total)")
        
        for i in range(page_limit):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            
            # Extract tables
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            }) or []
            
            # Extract images
            images = []
            pymupdf_page = pymupdf_doc[i]
            for img_idx, img in enumerate(pymupdf_page.get_images(full=True)):
                try:
                    base_image = pymupdf_doc.extract_image(img[0])
                    img_filename = f"page_{i+1:03d}_img_{img_idx+1}.{base_image['ext']}"
                    img_path = figures_dir / img_filename
                    img_path.write_bytes(base_image["image"])
                    images.append({'filename': img_filename, 'page': i+1})
                except:
                    pass
            
            # Detect code blocks
            code_blocks = detect_code_blocks(text)
            
            pages_data.append({
                'page_num': i + 1,
                'text': text,
                'tables': len(tables),
                'images': len(images),
                'code_blocks': len(code_blocks),
                'char_count': len(text)
            })
            
            if (i + 1) % 5 == 0:
                logging.info(f"✓ Processed page {i+1}/{page_limit}")
    
    pymupdf_doc.close()
    
    # Save parsed data
    output_file = output_dir / "pages_data.json"
    with open(output_file, 'w') as f:
        json.dump(pages_data, f, indent=2)
    
    logging.info(f"✓ Parsed {len(pages_data)} pages")
    logging.info(f"  Total chars: {sum(p['char_count'] for p in pages_data):,}")
    logging.info(f"  Total images: {sum(p['images'] for p in pages_data)}")
    logging.info(f"  Total code blocks: {sum(p['code_blocks'] for p in pages_data)}")
    
    # Pass stats to next task
    context['ti'].xcom_push(key='num_pages', value=len(pages_data))
    
    return "parse_complete"


def chunk_documents_task(**context):
    """
    Chunk documents using code-aware strategy
    Adapted from your chunk_documents.py - strategy 6
    """
    import json
    from pathlib import Path
    
    logging.info("="*70)
    logging.info("TASK 2: DOCUMENT CHUNKING")
    logging.info("="*70)
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
    except ImportError as e:
        logging.error(f"Missing library: {e}")
        raise
    
    # Load parsed pages
    input_file = Path("/tmp/processed/pages_data.json")
    with open(input_file, 'r') as f:
        pages_data = json.load(f)
    
    logging.info(f"Loaded {len(pages_data)} pages")
    
    # Convert to LangChain Documents
    documents = []
    for page in pages_data:
        doc = Document(
            page_content=page['text'],
            metadata={
                'source': 'fintbx.pdf',
                'page': page['page_num']
            }
        )
        documents.append(doc)
    
    # Code-aware chunking strategy (your strategy 6)
    code_separators = [
        "\n```\n",      # End of code blocks
        "\n### ",       # Section headers
        "\n## ",
        "\n**",         # Bold text
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
    
    # Chunk all documents
    chunks = splitter.split_documents(documents)
    
    # Convert to JSON format
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            'chunk_id': i,
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'char_count': len(chunk.page_content)
        })
    
    # Save chunks
    output_dir = Path("/tmp/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks.json"
    
    with open(output_file, 'w') as f:
        json.dump(chunks_data, f, indent=2)
    
    logging.info(f"✓ Created {len(chunks_data)} chunks")
    logging.info(f"  Avg size: {sum(c['char_count'] for c in chunks_data) / len(chunks_data):.0f} chars")
    
    context['ti'].xcom_push(key='num_chunks', value=len(chunks_data))
    
    return "chunk_complete"


def generate_embeddings_task(**context):
    """
    Generate embeddings using OpenAI text-embedding-3-large
    Adapted from your generate_embeddings.py
    """
    import json
    from pathlib import Path
    from openai import OpenAI
    
    logging.info("="*70)
    logging.info("TASK 3: EMBEDDING GENERATION")
    logging.info("="*70)
    
    # Get API key from Airflow Variable
    api_key = Variable.get("openai_api_key")
    client = OpenAI(api_key=api_key)
    
    # Load chunks
    chunks_file = Path("/tmp/chunks/chunks.json")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    logging.info(f"Loaded {len(chunks)} chunks")
    logging.info(f"Model: text-embedding-3-large")
    
    # Generate embeddings in batches
    embedded_chunks = []
    batch_size = 100
    total_batches = (len(chunks) - 1) // batch_size + 1
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c['content'] for c in batch]
        
        logging.info(f"Embedding batch {i//batch_size + 1}/{total_batches}...")
        
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        
        for chunk, embedding_obj in zip(batch, response.data):
            chunk['embedding'] = embedding_obj.embedding
            chunk['embedding_model'] = 'text-embedding-3-large'
            chunk['embedding_dimensions'] = 3072
            embedded_chunks.append(chunk)
        
        logging.info(f"  ✓ Batch {i//batch_size + 1}/{total_batches} complete")
    
    # Save embeddings
    output_dir = Path("/tmp/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "embeddings.json"
    
    with open(output_file, 'w') as f:
        json.dump(embedded_chunks, f, indent=2)
    
    # Save metadata
    metadata_file = output_dir / "embedding_metadata.json"
    metadata = {
        'num_embeddings': len(embedded_chunks),
        'model': 'text-embedding-3-large',
        'dimensions': 3072,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"✓ Generated {len(embedded_chunks)} embeddings")
    logging.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    context['ti'].xcom_push(key='num_embeddings', value=len(embedded_chunks))
    
    return "embeddings_complete"


# Define the DAG
with DAG(
    dag_id='fintbx_ingest_pipeline',
    default_args=default_args,
    description='Weekly ingestion pipeline for Financial Toolbox PDF',
    schedule_interval='0 0 * * 0',  # Every Sunday at midnight (weekly)
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=['production', 'weekly', 'ingestion', 'lab2'],
    doc_md="""
    # Financial Toolbox Ingestion Pipeline
    
    This DAG orchestrates the complete ingestion workflow:
    1. Download fintbx.pdf from GCS
    2. Parse PDF (text, tables, figures, code, formulas)
    3. Chunk documents using code-aware strategy
    4. Generate embeddings with text-embedding-3-large
    5. Upload all artifacts to GCS
    
    **Schedule:** Weekly (every Sunday at midnight UTC)
    **Duration:** ~10-15 minutes per run
    """
) as dag:
    
    # ========================================================================
    # TASK 1: Download PDF from GCS
    # ========================================================================
    download_pdf = GCSToLocalFilesystemOperator(
        task_id='download_pdf_from_gcs',
        bucket=GCS_BUCKET,
        object_name='raw_pdfs/fintbx.pdf',
        filename='/tmp/fintbx.pdf',
        doc_md="Downloads source PDF from Google Cloud Storage"
    )
    
    # ========================================================================
    # TASK 2: Parse PDF
    # ========================================================================
    parse_pdf = PythonOperator(
        task_id='parse_pdf',
        python_callable=parse_pdf_task,
        provide_context=True,
        doc_md="Extracts text, code, tables, figures using pdfplumber + PyMuPDF"
    )
    
    # ========================================================================
    # TASK 3: Chunk Documents
    # ========================================================================
    chunk_docs = PythonOperator(
        task_id='chunk_documents',
        python_callable=chunk_documents_task,
        provide_context=True,
        doc_md="Chunks text using code-aware RecursiveCharacterTextSplitter (1200/200)"
    )
    
    # ========================================================================
    # TASK 4: Generate Embeddings
    # ========================================================================
    embed_chunks = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings_task,
        provide_context=True,
        doc_md="Generates embeddings using OpenAI text-embedding-3-large (3072 dims)"
    )
    
    # ========================================================================
    # TASK 5: Upload Results to GCS
    # ========================================================================
    upload_processed = LocalFilesystemToGCSOperator(
        task_id='upload_processed_data',
        src='/tmp/processed/pages_data.json',
        dst='processed/{{ ds }}/pages_data.json',  # Date-stamped folder
        bucket=GCS_BUCKET,
        doc_md="Uploads parsed page data to GCS"
    )
    
    upload_chunks = LocalFilesystemToGCSOperator(
        task_id='upload_chunks',
        src='/tmp/chunks/chunks.json',
        dst='chunks/{{ ds }}/chunks.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads chunked documents to GCS"
    )
    
    upload_embeddings = LocalFilesystemToGCSOperator(
        task_id='upload_embeddings',
        src='/tmp/embeddings/embeddings.json',
        dst='embeddings/{{ ds }}/embeddings.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads generated embeddings to GCS"
    )
    
    upload_metadata = LocalFilesystemToGCSOperator(
        task_id='upload_metadata',
        src='/tmp/embeddings/embedding_metadata.json',
        dst='embeddings/{{ ds }}/metadata.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads embedding metadata to GCS"
    )
    
    # ========================================================================
    # TASK DEPENDENCIES (Execution Order)
    # ========================================================================
    
    # Main pipeline flow
    download_pdf >> parse_pdf >> chunk_docs >> embed_chunks
    
    # Upload results in parallel after each stage
    parse_pdf >> upload_processed
    chunk_docs >> upload_chunks
    embed_chunks >> [upload_embeddings, upload_metadata]
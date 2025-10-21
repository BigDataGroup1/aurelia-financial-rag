"""
Lab 2: concept_seed_dag - Concept Database Seeding
Generates structured JSON using instructor + GPT-4
Uploads outputs internally to avoid multi-worker issues
Schedule: On-demand (manual trigger)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import logging

# Configuration
GCS_BUCKET = Variable.get("gcs_bucket", default_var="aurelia-rag-data")

# Financial concepts to seed
FINANCIAL_CONCEPTS = [
    "Duration", "Convexity", "Sharpe Ratio", "Black-Scholes Model",
    "Present Value", "Future Value", "Yield to Maturity",
    "Modified Duration", "Macaulay Duration", "Bond Pricing",
    "Internal Rate of Return", "Net Present Value", "Discount Rate",
    "Coupon Rate", "Par Value", "Credit Spread", "Term Structure",
    "Spot Rate", "Forward Rate", "Option Greeks",
]

default_args = {
    'owner': 'aurelia-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def generate_and_upload_concepts_task(**context):
    """
    Complete concept generation pipeline with upload
    All in one task to avoid worker isolation
    """
    import json
    from pathlib import Path
    from openai import OpenAI
    from pydantic import BaseModel, Field
    from typing import List
    from google.cloud import storage
    import instructor
    
    logging.info("="*70)
    logging.info("CONCEPT GENERATION & UPLOAD")
    logging.info("="*70)
    
    # Get API key
    api_key = Variable.get("openai_api_key")
    
    # Create instructor client
    client = instructor.from_openai(OpenAI(api_key=api_key))
    
    # Define schema
    class ConceptDefinition(BaseModel):
        """Structured financial concept definition"""
        name: str = Field(description="The concept name")
        definition: str = Field(description="Clear, concise definition (2-3 sentences)")
        formula: str = Field(description="Mathematical formula if applicable, else 'N/A'")
        matlab_function: str = Field(description="Relevant MATLAB function from Financial Toolbox, if any")
        use_case: str = Field(description="Primary use case in finance")
        example: str = Field(description="Practical numerical example")
        related_concepts: List[str] = Field(description="List of 2-4 related concepts")
        category: str = Field(description="Category: risk, valuation, derivatives, or rates")
    
    # STEP 1: Generate definitions
    logging.info(f"Generating {len(FINANCIAL_CONCEPTS)} concepts...")
    
    definitions = []
    for idx, concept_name in enumerate(FINANCIAL_CONCEPTS):
        try:
            logging.info(f"  [{idx+1}/{len(FINANCIAL_CONCEPTS)}] {concept_name}")
            
            definition = client.chat.completions.create(
                model="gpt-4",
                response_model=ConceptDefinition,
                messages=[{
                    "role": "system",
                    "content": "You are a financial expert. Provide structured, accurate definitions."
                }, {
                    "role": "user",
                    "content": f"Define: {concept_name}"
                }],
                max_retries=2
            )
            
            definitions.append(definition.model_dump())
            logging.info(f"    ✓ Generated")
            
        except Exception as e:
            logging.warning(f"    ✗ Failed: {e}")
            continue
    
    # STEP 2: Validate
    logging.info("\nValidating concepts...")
    
    valid_count = 0
    warnings = []
    
    for defn in definitions:
        issues = []
        
        if not defn.get('definition') or len(defn['definition']) < 20:
            issues.append("Definition too short")
        if not defn.get('use_case'):
            issues.append("Missing use case")
        if not defn.get('category'):
            issues.append("Missing category")
        if not defn.get('related_concepts') or len(defn['related_concepts']) < 2:
            issues.append("Insufficient related concepts")
        
        if issues:
            warnings.append({'concept': defn['name'], 'issues': issues})
        else:
            valid_count += 1
    
    validation_report = {
        'total_concepts': len(definitions),
        'valid_concepts': valid_count,
        'validation_warnings': warnings,
        'validated_at': datetime.now().isoformat(),
        'validation_rate': (valid_count / len(definitions) * 100) if definitions else 0
    }
    
    logging.info(f"✓ Validation: {valid_count}/{len(definitions)} valid ({validation_report['validation_rate']:.1f}%)")
    
    # STEP 3: Create index
    logging.info("\nCreating concept index...")
    
    index = {
        'concepts': {},
        'by_category': {},
        'by_formula': {},
        'cross_references': {},
        'metadata': {
            'total_concepts': len(definitions),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0',
            'source': 'instructor_generated'
        }
    }
    
    for defn in definitions:
        name = defn['name']
        index['concepts'][name] = defn
        
        category = defn.get('category', 'uncategorized')
        if category not in index['by_category']:
            index['by_category'][category] = []
        index['by_category'][category].append(name)
        
        has_formula = defn.get('formula', 'N/A') != 'N/A'
        formula_key = 'with_formula' if has_formula else 'no_formula'
        if formula_key not in index['by_formula']:
            index['by_formula'][formula_key] = []
        index['by_formula'][formula_key].append(name)
        
        index['cross_references'][name] = defn.get('related_concepts', [])
    
    logging.info(f"✓ Index created: {len(index['by_category'])} categories")
    
    # STEP 4: Save locally
    output_dir = Path("/tmp/concepts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    definitions_file = output_dir / "concept_definitions.json"
    with open(definitions_file, 'w') as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)
    
    validation_file = output_dir / "validation_report.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    index_file = output_dir / "concept_index.json"
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    # STEP 5: Upload to GCS (same worker!)
    logging.info("\nUploading to GCS...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(Variable.get("gcs_bucket"))
    ds = context['ds']
    
    # Upload definitions
    blob = bucket.blob(f"concepts/{ds}/concept_definitions.json")
    blob.upload_from_filename(str(definitions_file))
    logging.info("✓ Uploaded concept_definitions.json")
    
    # Upload validation report
    blob = bucket.blob(f"concepts/{ds}/validation_report.json")
    blob.upload_from_filename(str(validation_file))
    logging.info("✓ Uploaded validation_report.json")
    
    # Upload index
    blob = bucket.blob(f"concepts/{ds}/concept_index.json")
    blob.upload_from_filename(str(index_file))
    logging.info("✓ Uploaded concept_index.json")
    
    context['ti'].xcom_push(key='num_concepts', value=len(definitions))
    context['ti'].xcom_push(key='valid_concepts', value=valid_count)
    
    return "concepts_complete"


# Define the DAG
with DAG(
    dag_id='concept_seed_pipeline',
    default_args=default_args,
    description='On-demand concept seeding with instructor + GPT-4',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=['concepts', 'on-demand', 'seeding', 'lab2'],
    doc_md="""
    # Concept Database Seeding Pipeline
    
    Single task that:
    1. Generates structured definitions (instructor + GPT-4)
    2. Validates completeness
    3. Creates searchable index
    4. Uploads all outputs to GCS
    
    **Schedule:** On-demand (manual trigger)
    **Concepts:** 20 financial terms
    **Duration:** ~5-10 minutes
    """
) as dag:
    
    # Single comprehensive task
    generate_concepts = PythonOperator(
        task_id='generate_and_upload_concepts',
        python_callable=generate_and_upload_concepts_task,
        provide_context=True,
    )
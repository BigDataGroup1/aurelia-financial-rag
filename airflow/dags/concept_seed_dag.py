"""
Lab 2: concept_seed_dag - Concept Database Seeding
Generates structured JSON for financial concepts using instructor
Schedule: On-demand (manual trigger)
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.models import Variable
import logging

# Configuration
GCS_BUCKET = Variable.get("gcs_bucket", default_var="your-bucket-name")

# Financial concepts to seed (as per assignment)
FINANCIAL_CONCEPTS = [
    "Duration",
    "Convexity", 
    "Sharpe Ratio",
    "Black-Scholes Model",
    "Present Value",
    "Future Value",
    "Yield to Maturity",
    "Modified Duration",
    "Macaulay Duration",
    "Bond Pricing",
    "Internal Rate of Return",
    "Net Present Value",
    "Discount Rate",
    "Coupon Rate",
    "Par Value",
    "Credit Spread",
    "Term Structure",
    "Spot Rate",
    "Forward Rate",
    "Option Greeks",
]

default_args = {
    'owner': 'aurelia-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def generate_concept_definitions_task(**context):
    """
    Generate structured concept definitions using instructor + OpenAI
    Uses Pydantic schemas for structured output
    """
    import json
    from pathlib import Path
    from openai import OpenAI
    from pydantic import BaseModel, Field
    from typing import List
    
    try:
        import instructor
    except ImportError:
        logging.error("instructor library not installed")
        logging.error("Add to PyPI packages: instructor==0.6.0")
        raise
    
    logging.info("="*70)
    logging.info("TASK 1: GENERATE CONCEPT DEFINITIONS")
    logging.info("="*70)
    
    # Get API key
    api_key = Variable.get("openai_api_key")
    
    # Create instructor client
    client = instructor.from_openai(OpenAI(api_key=api_key))
    
    # Define schema for structured output
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
    
    logging.info(f"Generating definitions for {len(FINANCIAL_CONCEPTS)} concepts...")
    
    # Generate definitions
    definitions = []
    for idx, concept_name in enumerate(FINANCIAL_CONCEPTS):
        try:
            logging.info(f"  [{idx+1}/{len(FINANCIAL_CONCEPTS)}] Generating: {concept_name}")
            
            # Use instructor to get structured output
            definition = client.chat.completions.create(
                model="gpt-4",
                response_model=ConceptDefinition,
                messages=[{
                    "role": "system",
                    "content": "You are a financial expert. Provide structured, accurate definitions for financial concepts."
                }, {
                    "role": "user",
                    "content": f"Provide a structured definition for the financial concept: {concept_name}"
                }],
                max_retries=2
            )
            
            definitions.append(definition.model_dump())
            logging.info(f"    ✓ {concept_name}")
            
        except Exception as e:
            logging.warning(f"    ✗ Failed to define {concept_name}: {e}")
            # Continue with others
            continue
    
    # Save definitions
    output_dir = Path("/tmp/concepts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "concept_definitions.json"
    
    with open(output_file, 'w') as f:
        json.dump(definitions, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n✓ Generated {len(definitions)} concept definitions")
    
    context['ti'].xcom_push(key='num_concepts', value=len(definitions))
    
    return "definitions_complete"


def validate_concepts_task(**context):
    """
    Validate concept definitions for completeness and quality
    """
    import json
    from pathlib import Path
    
    logging.info("="*70)
    logging.info("TASK 2: VALIDATE CONCEPTS")
    logging.info("="*70)
    
    # Load definitions
    defs_file = Path("/tmp/concepts/concept_definitions.json")
    with open(defs_file, 'r') as f:
        definitions = json.load(f)
    
    # Validation rules
    valid_count = 0
    warnings = []
    
    for defn in definitions:
        issues = []
        
        # Check required fields
        if not defn.get('definition') or len(defn['definition']) < 20:
            issues.append("Definition too short (<20 chars)")
        
        if not defn.get('use_case'):
            issues.append("Missing use case")
        
        if not defn.get('category'):
            issues.append("Missing category")
        
        if not defn.get('related_concepts') or len(defn['related_concepts']) < 2:
            issues.append("Insufficient related concepts (<2)")
        
        if issues:
            warnings.append({
                'concept': defn['name'],
                'issues': issues
            })
        else:
            valid_count += 1
    
    # Create validation report
    report = {
        'total_concepts': len(definitions),
        'valid_concepts': valid_count,
        'invalid_concepts': len(warnings),
        'validation_warnings': warnings,
        'validated_at': datetime.now().isoformat(),
        'validation_rate': (valid_count / len(definitions) * 100) if definitions else 0
    }
    
    report_file = Path("/tmp/concepts/validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"✓ Validation complete")
    logging.info(f"  Valid: {valid_count}/{len(definitions)} ({report['validation_rate']:.1f}%)")
    
    if warnings:
        logging.warning(f"  Found {len(warnings)} concepts with issues")
        for w in warnings[:3]:  # Show first 3
            logging.warning(f"    - {w['concept']}: {', '.join(w['issues'])}")
    
    context['ti'].xcom_push(key='valid_concepts', value=valid_count)
    
    return "validation_complete"


def create_concept_index_task(**context):
    """
    Create searchable concept index with cross-references
    """
    import json
    from pathlib import Path
    
    logging.info("="*70)
    logging.info("TASK 3: CREATE CONCEPT INDEX")
    logging.info("="*70)
    
    # Load definitions
    defs_file = Path("/tmp/concepts/concept_definitions.json")
    with open(defs_file, 'r') as f:
        definitions = json.load(f)
    
    # Build index structures
    index = {
        'concepts': {},  # name -> full definition
        'by_category': {},  # category -> [concept names]
        'by_formula': {},  # formula presence -> [concept names]
        'cross_references': {},  # concept -> related concepts
        'metadata': {
            'total_concepts': len(definitions),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0',
            'source': 'fintbx.pdf'
        }
    }
    
    for defn in definitions:
        name = defn['name']
        
        # Main index
        index['concepts'][name] = defn
        
        # Category index
        category = defn.get('category', 'uncategorized')
        if category not in index['by_category']:
            index['by_category'][category] = []
        index['by_category'][category].append(name)
        
        # Formula index
        has_formula = defn.get('formula', 'N/A') != 'N/A'
        formula_key = 'with_formula' if has_formula else 'no_formula'
        if formula_key not in index['by_formula']:
            index['by_formula'][formula_key] = []
        index['by_formula'][formula_key].append(name)
        
        # Cross-references
        index['cross_references'][name] = defn.get('related_concepts', [])
    
    # Save index
    index_file = Path("/tmp/concepts/concept_index.json")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    logging.info(f"✓ Created concept index")
    logging.info(f"  Total concepts: {len(definitions)}")
    logging.info(f"  Categories: {len(index['by_category'])}")
    logging.info(f"  With formulas: {len(index['by_formula'].get('with_formula', []))}")
    logging.info(f"  Without formulas: {len(index['by_formula'].get('no_formula', []))}")
    
    return "index_complete"


# Define the DAG
with DAG(
    dag_id='concept_seed_pipeline',
    default_args=default_args,
    description='On-demand concept database seeding with instructor',
    schedule_interval=None,  # Manual trigger only (on-demand)
    start_date=datetime(2025, 10, 1),
    catchup=False,
    tags=['concepts', 'on-demand', 'seeding', 'lab2'],
    doc_md="""
    # Concept Database Seeding Pipeline
    
    This DAG generates structured concept definitions:
    1. Generate definitions using instructor + GPT-4
    2. Validate definitions for completeness
    3. Create searchable index with cross-references
    4. Upload to GCS
    
    **Schedule:** On-demand (manual trigger)
    **Concepts:** 20 core financial terms
    **Duration:** ~5-10 minutes per run
    """
) as dag:
    
    # ========================================================================
    # TASK 1: Generate Structured Definitions
    # ========================================================================
    generate_definitions = PythonOperator(
        task_id='generate_concept_definitions',
        python_callable=generate_concept_definitions_task,
        provide_context=True,
        doc_md="Uses instructor library to generate Pydantic-validated concept definitions"
    )
    
    # ========================================================================
    # TASK 2: Validate Concepts
    # ========================================================================
    validate_concepts = PythonOperator(
        task_id='validate_concepts',
        python_callable=validate_concepts_task,
        provide_context=True,
        doc_md="Validates completeness: definition length, required fields, related concepts"
    )
    
    # ========================================================================
    # TASK 3: Create Searchable Index
    # ========================================================================
    create_index = PythonOperator(
        task_id='create_concept_index',
        python_callable=create_concept_index_task,
        provide_context=True,
        doc_md="Builds category index, formula index, and cross-reference graph"
    )
    
    # ========================================================================
    # TASK 4: Upload to GCS
    # ========================================================================
    upload_definitions = LocalFilesystemToGCSOperator(
        task_id='upload_concept_definitions',
        src='/tmp/concepts/concept_definitions.json',
        dst='concepts/{{ ds }}/definitions.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads concept definitions to GCS"
    )
    
    upload_validation = LocalFilesystemToGCSOperator(
        task_id='upload_validation_report',
        src='/tmp/concepts/validation_report.json',
        dst='concepts/{{ ds }}/validation_report.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads validation report to GCS"
    )
    
    upload_index = LocalFilesystemToGCSOperator(
        task_id='upload_concept_index',
        src='/tmp/concepts/concept_index.json',
        dst='concepts/{{ ds }}/index.json',
        bucket=GCS_BUCKET,
        doc_md="Uploads searchable concept index to GCS"
    )
    
    # ========================================================================
    # TASK DEPENDENCIES (Execution Order)
    # ========================================================================
    
    generate_definitions >> validate_concepts >> create_index >> [
        upload_definitions,
        upload_validation,
        upload_index
    ]
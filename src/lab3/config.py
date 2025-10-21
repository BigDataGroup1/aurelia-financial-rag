"""
Configuration Management for Lab 3 RAG Service
Uses pydantic-settings for type-safe configuration from environment variables
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # GCS Configuration  
    gcs_bucket: str = Field(default="aurelia-rag-data")
    gcp_project_id: str = Field(default="aurelia-financial-rag")

    # Paths (App Engine uses /tmp)
    chromadb_path: Path = Field(default=Path("/tmp/chromadb"))
        
    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")
    
    # OpenAI Configuration
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for generation"
    )
    embedding_dimensions: int = Field(
        default=3072,
        description="Embedding dimensions for text-embedding-3-large"
    )
    
    # ChromaDB Configuration
    chromadb_collection_name: str = Field(
        default="fintbx",
        description="ChromaDB collection name"
    )
    
    # RAG Configuration
    top_k_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from vector store"
    )
    similarity_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for using PDF chunks (0-1)"
    )
    max_context_tokens: int = Field(
        default=6000,
        description="Maximum tokens for context assembly"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./concept_notes.db",
        description="Database connection URL"
    )
    
    # API Configuration
    api_title: str = "AURELIA Financial RAG Service"
    api_version: str = "1.0.0"
    api_description: str = """
    Production-grade RAG service for financial concept note generation.
    
    Features:
    - Vector-based retrieval from Financial Toolbox
    - Wikipedia fallback for unknown concepts
    - Structured output with citations
    - Intelligent caching for fast queries
    """
    
    # Wikipedia Configuration
    wikipedia_sentences: int = Field(
        default=10,
        description="Number of sentences to fetch from Wikipedia summary"
    )
    wikipedia_content_limit: int = Field(
        default=2000,
        description="Character limit for Wikipedia page content"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


# Create global settings instance
settings = Settings()


# Validate critical paths on import
if not settings.chromadb_path.exists():
    raise RuntimeError(
        f"ChromaDB path not found: {settings.chromadb_path}\n"
        "Please ensure Lab 1 has been completed and ChromaDB data exists."
    )

print(f"✓ Configuration loaded successfully")
print(f"  - ChromaDB: {settings.chromadb_path}")
print(f"  - Database: {settings.database_url}")
print(f"  - LLM Model: {settings.llm_model}")
print(f"  - Similarity Threshold: {settings.similarity_threshold}")


if __name__ == "__main__":
    # Test configuration
    print("\n" + "="*60)
    print("CONFIGURATION TEST")
    print("="*60)
    print(f"Project Root: {settings.project_root}")
    print(f"ChromaDB Path: {settings.chromadb_path}")
    print(f"ChromaDB Exists: {settings.chromadb_path.exists()}")
    print(f"OpenAI API Key: {settings.openai_api_key[:10]}...{settings.openai_api_key[-4:]}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Top K Results: {settings.top_k_results}")
    print(f"Similarity Threshold: {settings.similarity_threshold}")
    print("="*60)
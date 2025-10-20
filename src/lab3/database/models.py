"""
SQLAlchemy database models for caching concept notes
"""
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

from ..config import settings

Base = declarative_base()


class ConceptNoteCache(Base):
    """
    Database model for cached concept notes
    """
    __tablename__ = 'concept_notes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    concept_name = Column(String(255), unique=True, nullable=False, index=True)
    
    # ConceptNote fields
    definition = Column(String, nullable=False)
    key_components = Column(JSON, nullable=False)
    formula = Column(String, nullable=True)
    example = Column(String, nullable=False)
    use_cases = Column(JSON, nullable=False)
    related_concepts = Column(JSON, nullable=False)
    
    # Metadata
    source = Column(String(50), nullable=False)
    page_references = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ConceptNoteCache(concept='{self.concept_name}', source='{self.source}')>"


# Database setup
def get_engine():
    """Create database engine"""
    # Ensure database directory exists for SQLite
    if settings.database_url.startswith('sqlite'):
        # Extract path from sqlite:///./concept_notes.db
        db_path = settings.database_url.replace('sqlite:///', '')
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    engine = create_engine(
        settings.database_url,
        echo=False,  # Set to True for SQL debugging
        connect_args={"check_same_thread": False} if 'sqlite' in settings.database_url else {}
    )
    return engine


def init_database():
    """Initialize database - create tables if they don't exist"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


# Session factory
def get_session_factory():
    """Get SQLAlchemy session factory"""
    engine = init_database()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal


# Global session factory
_session_factory = None


def get_session():
    """Get a database session"""
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory()
    return _session_factory()
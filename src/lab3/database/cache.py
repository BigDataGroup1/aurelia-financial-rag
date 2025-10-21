"""
Cache service for storing and retrieving concept notes
"""
from typing import Optional
import logging
from datetime import datetime

from ..models import ConceptNote
from .models import ConceptNoteCache, get_session

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching concept notes in database"""
    
    def __init__(self):
        logger.info("CacheService initialized")
    
    def get(self, concept_name: str) -> Optional[ConceptNote]:
        """
        Retrieve a cached concept note
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            ConceptNote if found in cache, None otherwise
        """
        try:
            session = get_session()
            
            # Query database
            cached = session.query(ConceptNoteCache).filter(
                ConceptNoteCache.concept_name == concept_name.strip()
            ).first()
            
            session.close()
            
            if cached:
                logger.info(f"Cache HIT for '{concept_name}'")
                
                # Convert database model to Pydantic model
                concept_note = ConceptNote(
                    concept_name=cached.concept_name,
                    definition=cached.definition,
                    key_components=cached.key_components,
                    formula=cached.formula,
                    example=cached.example,
                    use_cases=cached.use_cases,
                    related_concepts=cached.related_concepts,
                    source=cached.source,
                    page_references=cached.page_references,
                    confidence=cached.confidence
                )
                
                return concept_note
            else:
                logger.info(f"Cache MISS for '{concept_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Cache retrieval error for '{concept_name}': {e}")
            return None
    
    def save(self, concept_note: ConceptNote) -> bool:
        """
        Save a concept note to cache
        
        Args:
            concept_note: ConceptNote to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = get_session()
            
            # Check if already exists
            existing = session.query(ConceptNoteCache).filter(
                ConceptNoteCache.concept_name == concept_note.concept_name
            ).first()
            
            if existing:
                # Update existing
                logger.info(f"Updating cached note for '{concept_note.concept_name}'")
                existing.definition = concept_note.definition
                existing.key_components = concept_note.key_components
                existing.formula = concept_note.formula
                existing.example = concept_note.example
                existing.use_cases = concept_note.use_cases
                existing.related_concepts = concept_note.related_concepts
                existing.source = concept_note.source
                existing.page_references = concept_note.page_references
                existing.confidence = concept_note.confidence
                existing.updated_at = datetime.utcnow()
            else:
                # Create new
                logger.info(f"Caching new note for '{concept_note.concept_name}'")
                cached = ConceptNoteCache(
                    concept_name=concept_note.concept_name,
                    definition=concept_note.definition,
                    key_components=concept_note.key_components,
                    formula=concept_note.formula,
                    example=concept_note.example,
                    use_cases=concept_note.use_cases,
                    related_concepts=concept_note.related_concepts,
                    source=concept_note.source,
                    page_references=concept_note.page_references,
                    confidence=concept_note.confidence
                )
                session.add(cached)
            
            session.commit()
            session.close()
            
            logger.info(f"✓ Successfully cached '{concept_note.concept_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Cache save error for '{concept_note.concept_name}': {e}")
            return False
    
    def delete(self, concept_name: str) -> bool:
        """Delete a cached concept note"""
        try:
            session = get_session()
            
            deleted = session.query(ConceptNoteCache).filter(
                ConceptNoteCache.concept_name == concept_name
            ).delete()
            
            session.commit()
            session.close()
            
            if deleted:
                logger.info(f"Deleted cached note for '{concept_name}'")
                return True
            else:
                logger.warning(f"No cached note found for '{concept_name}'")
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for '{concept_name}': {e}")
            return False
    
    def count(self) -> int:
        """Get count of cached concepts"""
        try:
            session = get_session()
            count = session.query(ConceptNoteCache).count()
            session.close()
            return count
        except:
            return 0
    
    def health_check(self) -> bool:
        """Check if database is accessible"""
        try:
            session = get_session()
            session.query(ConceptNoteCache).first()
            session.close()
            return True
        except:
            return False


# Global instance
_cache_service = None


def get_cache_service() -> CacheService:
    """Get or create the global CacheService instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
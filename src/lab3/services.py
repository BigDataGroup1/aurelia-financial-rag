"""
All RAG Services Combined
- EmbeddingService: OpenAI embeddings
- VectorStoreService: ChromaDB operations
- WikipediaService: Wikipedia fallback
- GenerationService: LLM generation with Instructor
"""
from typing import List, Dict, Optional
import logging

# OpenAI & Instructor
from openai import OpenAI
import instructor

# ChromaDB
import chromadb

# Wikipedia
import wikipedia

# Local imports
from config import settings
from models import ConceptNote, GenerationContext, RetrievalResult

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDING SERVICE
# ============================================================================

class EmbeddingService:
    """Service for generating embeddings with OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        logger.info(f"EmbeddingService initialized with model: {self.model}")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text
        
        Args:
            text: Query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            logger.debug(f"Embedding query: {text[:100]}...")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Validate dimensions
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Unexpected embedding dimensions: {len(embedding)} "
                    f"(expected {self.dimensions})"
                )
            
            logger.debug(f"Successfully generated {len(embedding)}D embedding")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            logger.debug(f"Embedding batch of {len(texts)} texts")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise Exception(f"Batch embedding generation failed: {str(e)}")


# ============================================================================
# VECTOR STORE SERVICE
# ============================================================================

"""
Replace VectorStoreService class in src/lab3/services.py
Around line 95-250
"""

class VectorStoreService:
    """Service for querying ChromaDB vector store"""
    
    def __init__(self, chromadb_path: str = None):
        """
        Initialize ChromaDB client and load collection
        
        Args:
            chromadb_path: Path to ChromaDB directory (optional)
                          If None, uses settings.chromadb_path
        """
        try:
            # Use provided path or default from settings
            if chromadb_path is None:
                chromadb_path = str(settings.chromadb_path)
            
            logger.info(f"Initializing ChromaDB from: {chromadb_path}")
            
            self.client = chromadb.PersistentClient(path=chromadb_path)
            self.collection = self.client.get_collection(
                name=settings.chromadb_collection_name
            )
            self.collection_size = self.collection.count()
            
            logger.info(
                f"VectorStoreService initialized: "
                f"collection='{settings.chromadb_collection_name}', "
                f"vectors={self.collection_size}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {str(e)}")
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = None,
        metadata_filter: Optional[Dict] = None
    ) -> RetrievalResult:
        """
        Query vector store with embedding
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (default from settings)
            metadata_filter: Optional metadata filters
            
        Returns:
            RetrievalResult with documents, metadata, and similarity scores
        """
        try:
            n_results = n_results or settings.top_k_results
            
            logger.debug(f"Querying ChromaDB: n_results={n_results}, filter={metadata_filter}")
            
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results * 2,  # Get extra for filtering
            }
            
            if metadata_filter:
                query_params["where"] = metadata_filter
            
            # Execute query
            results = self.collection.query(**query_params)
            
            # Post-process: convert distances to similarities and filter
            filtered_result = self._post_process_results(
                results,
                threshold=settings.similarity_threshold,
                max_results=n_results
            )
            
            if filtered_result.documents:
                logger.info(
                    f"Retrieved {len(filtered_result.documents)} chunks "
                    f"(avg similarity: {sum(filtered_result.similarities)/len(filtered_result.similarities):.3f})"
                )
            else:
                logger.warning("No results above threshold")
            
            return filtered_result
            
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise Exception(f"Vector store query failed: {str(e)}")
    
    def _post_process_results(
        self,
        raw_results: Dict,
        threshold: float,
        max_results: int
    ) -> RetrievalResult:
        """Post-process ChromaDB results: convert distances, filter, deduplicate"""
        
        if not raw_results['ids'][0]:
            return RetrievalResult(
                documents=[],
                metadatas=[],
                similarities=[],
                source='fintbx.pdf'
            )
        
        # Convert L2 distances to similarities: similarity = 1 / (1 + distance)
        distances = raw_results['distances'][0]
        similarities = [1 / (1 + d) for d in distances]
        
        # Filter by threshold
        good_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
        
        if not good_indices:
            logger.warning(
                f"No results above threshold {threshold}. "
                f"Best similarity: {max(similarities):.3f}"
            )
            return RetrievalResult(
                documents=[],
                metadatas=[],
                similarities=[],
                source='fintbx.pdf'
            )
        
        # Take top results
        good_indices = good_indices[:max_results]
        
        # Extract filtered results
        documents = [raw_results['documents'][0][i] for i in good_indices]
        metadatas = [raw_results['metadatas'][0][i] for i in good_indices]
        filtered_similarities = [similarities[i] for i in good_indices]
        
        # Deduplicate by content (keep highest similarity)
        seen_content = {}
        unique_docs = []
        unique_metas = []
        unique_sims = []
        
        for doc, meta, sim in zip(documents, metadatas, filtered_similarities):
            key = doc[:200]  # First 200 chars as dedup key
            if key not in seen_content or sim > seen_content[key]:
                if key in seen_content:
                    # Replace with higher similarity version
                    idx = unique_docs.index(next(d for d in unique_docs if d[:200] == key))
                    unique_docs[idx] = doc
                    unique_metas[idx] = meta
                    unique_sims[idx] = sim
                else:
                    unique_docs.append(doc)
                    unique_metas.append(meta)
                    unique_sims.append(sim)
                    seen_content[key] = sim
        
        logger.debug(f"Post-processing: {len(documents)} → {len(unique_docs)} (after deduplication)")
        
        return RetrievalResult(
            documents=unique_docs,
            metadatas=unique_metas,
            similarities=unique_sims,
            source='fintbx.pdf'
        )
    
    def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False


# Update the singleton getter to accept path parameter
_vector_store_service = None
def get_vector_store_service(chromadb_path: str = None) -> VectorStoreService:
    """Get or create the global VectorStoreService instance"""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService(chromadb_path)
    return _vector_store_service
    
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = None,
        metadata_filter: Optional[Dict] = None
    ) -> RetrievalResult:
        """
        Query vector store with embedding
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (default from settings)
            metadata_filter: Optional metadata filters
            
        Returns:
            RetrievalResult with documents, metadata, and similarity scores
        """
        try:
            n_results = n_results or settings.top_k_results
            
            logger.debug(f"Querying ChromaDB: n_results={n_results}, filter={metadata_filter}")
            
            # Build query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results * 2,  # Get extra for filtering
            }
            
            if metadata_filter:
                query_params["where"] = metadata_filter
            
            # Execute query
            results = self.collection.query(**query_params)
            
            # Post-process: convert distances to similarities and filter
            filtered_result = self._post_process_results(
                results,
                threshold=settings.similarity_threshold,
                max_results=n_results
            )
            
            if filtered_result.documents:
                logger.info(
                    f"Retrieved {len(filtered_result.documents)} chunks "
                    f"(avg similarity: {sum(filtered_result.similarities)/len(filtered_result.similarities):.3f})"
                )
            else:
                logger.warning("No results above threshold")
            
            return filtered_result
            
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise Exception(f"Vector store query failed: {str(e)}")
    
    def _post_process_results(
        self,
        raw_results: Dict,
        threshold: float,
        max_results: int
    ) -> RetrievalResult:
        """Post-process ChromaDB results: convert distances, filter, deduplicate"""
        
        if not raw_results['ids'][0]:
            return RetrievalResult(
                documents=[],
                metadatas=[],
                similarities=[],
                source='fintbx.pdf'
            )
        
        # Convert L2 distances to similarities: similarity = 1 / (1 + distance)
        distances = raw_results['distances'][0]
        similarities = [1 / (1 + d) for d in distances]
        
        # Filter by threshold
        good_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
        
        if not good_indices:
            logger.warning(
                f"No results above threshold {threshold}. "
                f"Best similarity: {max(similarities):.3f}"
            )
            return RetrievalResult(
                documents=[],
                metadatas=[],
                similarities=[],
                source='fintbx.pdf'
            )
        
        # Take top results
        good_indices = good_indices[:max_results]
        
        # Extract filtered results
        documents = [raw_results['documents'][0][i] for i in good_indices]
        metadatas = [raw_results['metadatas'][0][i] for i in good_indices]
        filtered_similarities = [similarities[i] for i in good_indices]
        
        # Deduplicate by content (keep highest similarity)
        seen_content = {}
        unique_docs = []
        unique_metas = []
        unique_sims = []
        
        for doc, meta, sim in zip(documents, metadatas, filtered_similarities):
            key = doc[:200]  # First 200 chars as dedup key
            if key not in seen_content or sim > seen_content[key]:
                if key in seen_content:
                    # Replace with higher similarity version
                    idx = unique_docs.index(next(d for d in unique_docs if d[:200] == key))
                    unique_docs[idx] = doc
                    unique_metas[idx] = meta
                    unique_sims[idx] = sim
                else:
                    unique_docs.append(doc)
                    unique_metas.append(meta)
                    unique_sims.append(sim)
                    seen_content[key] = sim
        
        logger.debug(f"Post-processing: {len(documents)} → {len(unique_docs)} (after deduplication)")
        
        return RetrievalResult(
            documents=unique_docs,
            metadatas=unique_metas,
            similarities=unique_sims,
            source='fintbx.pdf'
        )
    
    def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False


# ============================================================================
# WIKIPEDIA SERVICE
# ============================================================================

class WikipediaService:
    """Service for fetching content from Wikipedia as fallback"""
    
    def __init__(self):
        """Initialize Wikipedia service"""
        wikipedia.set_lang("en")
        logger.info("WikipediaService initialized")
    
    def search(self, concept: str) -> RetrievalResult:
        """
        Search Wikipedia for a concept and return content
        
        Args:
            concept: Concept name to search for
            
        Returns:
            RetrievalResult with Wikipedia content
        """
        try:
            logger.debug(f"Searching Wikipedia for: {concept}")
            
            # Try to get the page directly
            try:
                page = wikipedia.page(concept, auto_suggest=False)
                content = self._format_wikipedia_content(page)
                
                logger.info(f"Found Wikipedia page: '{page.title}' ({len(content)} chars)")
                
            except wikipedia.DisambiguationError as e:
                # Multiple possible pages - try to find best match
                logger.warning(
                    f"Disambiguation for '{concept}'. "
                    f"Options: {e.options[:5]}"
                )
    
                # Try to find option that includes the concept name (e.g., "Duration (finance)")
                best_option = e.options[0]
                for option in e.options[:5]:
                    if concept.lower() in option.lower():
                        best_option = option
                        break
    
                logger.info(f"Selected: {best_option}")
                page = wikipedia.page(best_option, auto_suggest=False)
                content = self._format_wikipedia_content(page)
                
            except wikipedia.PageError:
                # Page not found - try search
                logger.warning(f"No exact match for '{concept}'. Trying search...")
                search_results = wikipedia.search(concept, results=3)
                
                if not search_results:
                    raise Exception(f"No Wikipedia results found for '{concept}'")
                
                logger.info(f"Search results: {search_results}")
                page = wikipedia.page(search_results[0], auto_suggest=False)
                content = self._format_wikipedia_content(page)
            
            # Create RetrievalResult
            return RetrievalResult(
                documents=[content],
                metadatas=[{"source": "wikipedia", "title": page.title, "url": page.url}],
                similarities=[0.7],  # Fixed similarity for Wikipedia fallback
                source='wikipedia'
            )
            
        except Exception as e:
            logger.error(f"Wikipedia search failed for '{concept}': {e}")
            raise Exception(f"Wikipedia fallback failed: {str(e)}")
    
    def _format_wikipedia_content(self, page) -> str:
        """Format Wikipedia page content for concept note generation"""
        summary = page.summary
        content_limit = settings.wikipedia_content_limit
        full_content = page.content[:content_limit]
        
        formatted = f"""
# {page.title}

## Summary
{summary}

## Detailed Content
{full_content}

## Source
Wikipedia: {page.url}
        """.strip()
        
        return formatted
    
    def health_check(self) -> bool:
        """Test Wikipedia API connectivity"""
        try:
            results = wikipedia.search("Python", results=1)
            return len(results) > 0
        except:
            return False


# ============================================================================
# GENERATION SERVICE
# ============================================================================

class GenerationService:
    """Service for generating structured concept notes with LLM"""
    
    # System prompt for concept note generation
    SYSTEM_PROMPT = """You are an expert financial analyst and educator specializing in creating clear, accurate concept notes.

Your task is to generate a comprehensive, structured concept note based on the provided context.

Guidelines:
1. **Definition**: Provide a clear, concise 2-3 sentence definition that anyone can understand
2. **Key Components**: List 3-5 essential elements that make up this concept
3. **Formula**: Include the mathematical formula if applicable (use LaTeX notation)
4. **Example**: Give a concrete, practical example with specific numbers when possible
5. **Use Cases**: Describe 2-4 real-world applications in finance
6. **Related Concepts**: List related financial concepts (if any)
7. **Accuracy**: Only include information supported by the context provided
8. **Citations**: If context is from fintbx.pdf, note which pages are referenced

Be precise, factual, and educational. Avoid speculation or information not in the context."""
    
    def __init__(self):
        """Initialize LLM client with Instructor"""
    # For instructor >= 1.0.0
        import instructor
        self.client = instructor.patch(OpenAI(api_key=settings.openai_api_key))
        self.model = settings.llm_model
        logger.info(f"GenerationService initialized with model: {self.model}")
    
    def generate_concept_note(
        self,
        generation_context: GenerationContext
    ) -> ConceptNote:
        """
        Generate a structured concept note using LLM with Instructor
        
        Args:
            generation_context: Context containing concept, content, and metadata
            
        Returns:
            Validated ConceptNote object
        """
        try:
            logger.debug(f"Generating concept note for: {generation_context.concept}")
            
            # Construct user message with context
            user_message = self._build_user_message(generation_context)
            
            # Generate with Instructor (automatic validation against ConceptNote schema)
            concept_note = self.client.chat.completions.create(
                model=self.model,
                response_model=ConceptNote,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_retries=3,  # Instructor will retry on validation errors
            )
            
            # Set metadata from context
            concept_note.source = generation_context.source
            concept_note.page_references = generation_context.page_references
            concept_note.confidence = generation_context.confidence
            
            logger.info(
                f"Successfully generated concept note for '{generation_context.concept}' "
                f"(source: {concept_note.source}, confidence: {concept_note.confidence:.2f})"
            )
            
            return concept_note
            
        except Exception as e:
            logger.error(f"Failed to generate concept note: {e}")
            raise Exception(f"Concept note generation failed: {str(e)}")
    
    def _build_user_message(self, context: GenerationContext) -> str:
        """Build the user message with concept and context"""
        message_parts = [
            f"# Concept: {context.concept}",
            "",
            "## Context Information",
            f"Source: {context.source}",
        ]
        
        if context.page_references:
            message_parts.append(f"Pages: {', '.join(map(str, context.page_references))}")
        
        message_parts.extend([
            "",
            "## Relevant Content",
            context.context,
            "",
            "---",
            "",
            f"Please generate a comprehensive concept note for '{context.concept}' based on the context above.",
            "Ensure all information is accurate and supported by the provided content."
        ])
        
        return "\n".join(message_parts)
    
    def health_check(self) -> bool:
        """Test OpenAI API connectivity"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except:
            return False


# ============================================================================
# SERVICE INSTANCES (Singleton Pattern)
# ============================================================================

_embedding_service = None
_vector_store_service = None
_wikipedia_service = None
_generation_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global EmbeddingService instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service



def get_wikipedia_service() -> WikipediaService:
    """Get or create the global WikipediaService instance"""
    global _wikipedia_service
    if _wikipedia_service is None:
        _wikipedia_service = WikipediaService()
    return _wikipedia_service


def get_generation_service() -> GenerationService:
    """Get or create the global GenerationService instance"""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
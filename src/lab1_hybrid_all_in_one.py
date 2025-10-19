"""
AURELIA Lab 1 - Complete All-in-One Solution (Markdown-Based)
===============================================
Production-grade PDF processing with innovative features:
- PDF to Markdown conversion
- Hierarchical document understanding
- Multi-modal content extraction
- Markdown-aware semantic chunking
- Hybrid search (dense + sparse)
- Rich metadata enrichment

Key Innovation: Uses Markdown as intermediate format for better structure preservation!

Author: [Your Name]
Date: October 2025
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import statistics

# PDF Processing
import fitz  # PyMuPDF

# Environment
from dotenv import load_dotenv

# ML & Embeddings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Text Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# Vector Stores
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except (ImportError, Exception):
    PINECONE_AVAILABLE = False
    print("⚠️  Pinecone not available (using ChromaDB only)")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except (ImportError, Exception):
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Load environment variables
load_dotenv()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DocumentSection:
    """Represents a hierarchical section of the document"""
    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection
    page_start: int
    page_end: Optional[int]
    content: str
    parent_section: Optional[str]


@dataclass
class ExtractedElement:
    """Represents extracted non-text elements"""
    element_type: str  # 'formula', 'table', 'figure', 'code'
    content: Any
    page_number: int
    bbox: tuple
    caption: Optional[str]
    context: str


@dataclass
class SemanticChunk:
    """Enhanced chunk with metadata"""
    chunk_id: str
    content: str
    section_title: str
    section_hierarchy: List[str]
    page_numbers: List[int]
    chunk_type: str
    related_elements: List[str]
    metadata: Dict[str, Any]


# ============================================================================
# ADVANCED PDF PARSER WITH MARKDOWN CONVERSION
# ============================================================================

class AdvancedPDFParser:
    """
    Innovative PDF parser with:
    - PDF to Markdown conversion
    - Hierarchical structure extraction
    - Multi-modal element detection
    - Markdown-aware semantic chunking
    - Intelligent metadata enrichment
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.sections: List[DocumentSection] = []
        self.elements: List[ExtractedElement] = []
        self.chunks: List[SemanticChunk] = []
        self.markdown_content: str = ""
        
        # For semantic analysis
        print("🤖 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def convert_to_markdown(self) -> str:
        """
        Convert PDF to Markdown format
        Preserves structure with headers, code blocks, and formatting
        """
        print("📝 Converting PDF to Markdown...")
        
        markdown_lines = []
        markdown_lines.append(f"# {self.pdf_path.stem}\n\n")
        markdown_lines.append("---\n\n")
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            # Add page marker
            markdown_lines.append(f"\n<!-- Page {page_num + 1} -->\n\n")
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                is_code = False
                is_header = False
                font_sizes = []
                
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        is_bold = span["flags"] & 2 ** 4
                        is_italic = span["flags"] & 2 ** 1
                        
                        font_sizes.append(font_size)
                        
                        # Check if code
                        if "Courier" in span["font"] or "Mono" in span["font"]:
                            is_code = True
                        
                        # Check if header
                        if is_bold and font_size > 11 and len(text.split()) < 15:
                            is_header = True
                        
                        # Apply formatting
                        if is_bold and not is_header:
                            text = f"**{text}**"
                        if is_italic:
                            text = f"*{text}*"
                        
                        line_text += text
                    
                    block_text += line_text + "\n"
                
                block_text = block_text.strip()
                
                if not block_text:
                    continue
                
                # Format based on type
                if is_header:
                    # Determine header level from font size
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                    if avg_font_size >= 16:
                        markdown_lines.append(f"\n# {block_text}\n\n")
                    elif avg_font_size >= 14:
                        markdown_lines.append(f"\n## {block_text}\n\n")
                    else:
                        markdown_lines.append(f"\n### {block_text}\n\n")
                elif is_code:
                    markdown_lines.append(f"\n```\n{block_text}\n```\n\n")
                else:
                    # Regular paragraph
                    markdown_lines.append(f"{block_text}\n\n")
        
        self.markdown_content = "".join(markdown_lines)
        print(f"   ✓ Converted to Markdown ({len(self.markdown_content)} characters)")
        
        return self.markdown_content
    
    def chunk_markdown(self, chunk_size: int = 1000, overlap: int = 200) -> List[SemanticChunk]:
        """
        Chunk Markdown content using semantic similarity
        Respects Markdown structure (headers, code blocks, etc.)
        """
        print(f"✂️  Chunking Markdown (size={chunk_size}, overlap={overlap})...")
        
        if not self.markdown_content:
            self.convert_to_markdown()
        
        chunks = []
        
        # Split by headers to respect document structure
        sections = re.split(r'\n(#{1,3}\s+.+)\n', self.markdown_content)
        
        current_section_title = "Introduction"
        current_section_content = ""
        
        for i, part in enumerate(sections):
            # Check if this is a header
            if re.match(r'^#{1,3}\s+', part):
                # Process previous section
                if current_section_content.strip():
                    section_chunks = self._chunk_section_semantically(
                        current_section_content,
                        current_section_title,
                        chunk_size,
                        overlap,
                        len(chunks)
                    )
                    chunks.extend(section_chunks)
                
                # Start new section
                current_section_title = part.strip('#').strip()
                current_section_content = ""
            else:
                current_section_content += part
        
        # Process last section
        if current_section_content.strip():
            section_chunks = self._chunk_section_semantically(
                current_section_content,
                current_section_title,
                chunk_size,
                overlap,
                len(chunks)
            )
            chunks.extend(section_chunks)
        
        self.chunks = chunks
        print(f"   ✓ Created {len(chunks)} Markdown chunks")
        
        return chunks
    
    def _chunk_section_semantically(self, 
                                    content: str, 
                                    section_title: str,
                                    chunk_size: int,
                                    overlap: int,
                                    start_idx: int) -> List[SemanticChunk]:
        """
        Chunk a section using semantic similarity
        """
        # Split into paragraphs (respect Markdown structure)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return []
        
        # Get embeddings for paragraphs
        try:
            embeddings = self.sentence_model.encode(paragraphs)
        except:
            # Fallback to simple chunking if embedding fails
            return self._simple_chunk_section(content, section_title, chunk_size, overlap, start_idx)
        
        chunks = []
        current_chunk = []
        current_chunk_text = ""
        current_embedding = None
        
        for paragraph, embedding in zip(paragraphs, embeddings):
            if not current_chunk:
                current_chunk.append(paragraph)
                current_chunk_text = paragraph
                current_embedding = embedding
            else:
                # Check semantic similarity
                similarity = cosine_similarity([current_embedding], [embedding])[0][0]
                
                # Check if we should start new chunk
                if similarity < 0.5 or len(current_chunk_text) > chunk_size:
                    # Save current chunk
                    chunks.append(self._create_markdown_chunk(
                        content='\n\n'.join(current_chunk),
                        section_title=section_title,
                        chunk_index=start_idx + len(chunks)
                    ))
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > 1 and overlap > 0:
                        current_chunk = current_chunk[-1:]
                        current_chunk_text = current_chunk[0]
                    else:
                        current_chunk = []
                        current_chunk_text = ""
                
                current_chunk.append(paragraph)
                current_chunk_text += '\n\n' + paragraph
                
                # Update embedding (moving average)
                if current_embedding is not None:
                    current_embedding = np.mean([current_embedding, embedding], axis=0)
                else:
                    current_embedding = embedding
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_markdown_chunk(
                content='\n\n'.join(current_chunk),
                section_title=section_title,
                chunk_index=start_idx + len(chunks)
            ))
        
        return chunks
    
    def _simple_chunk_section(self, 
                             content: str, 
                             section_title: str,
                             chunk_size: int,
                             overlap: int,
                             start_idx: int) -> List[SemanticChunk]:
        """
        Fallback: Simple text-based chunking
        """
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(self._create_markdown_chunk(
                content=chunk_text,
                section_title=section_title,
                chunk_index=start_idx + len(chunks)
            ))
        
        return chunks
    
    def _create_markdown_chunk(self, 
                               content: str, 
                               section_title: str,
                               chunk_index: int) -> SemanticChunk:
        """
        Create a chunk object from Markdown content
        """
        # Classify chunk type based on Markdown markers
        chunk_type = self._classify_markdown_chunk(content)
        
        # Extract page number from content (if available)
        page_match = re.search(r'<!-- Page (\d+) -->', content)
        page_num = int(page_match.group(1)) if page_match else 0
        
        return SemanticChunk(
            chunk_id=f"md_chunk_{chunk_index:04d}",
            content=content,
            section_title=section_title,
            section_hierarchy=[section_title],
            page_numbers=[page_num],
            chunk_type=chunk_type,
            related_elements=[],
            metadata={
                'section_level': content.count('#'),
                'char_count': len(content),
                'word_count': len(content.split()),
                'has_code': '```' in content,
                'has_formula': bool(re.search(r'[=+\-*/]', content)),
            }
        )
    
    def _classify_markdown_chunk(self, content: str) -> str:
        """
        Classify chunk type based on Markdown markers
        """
        if '```' in content:
            return 'code'
        elif any(word in content.lower() for word in ['define', 'definition', 'is a']):
            return 'definition'
        elif any(word in content.lower() for word in ['example', 'for instance']):
            return 'example'
        elif re.search(r'\$.*\$|[A-Z]\s*=\s*', content):
            return 'formula'
        else:
            return 'explanation'
    
    def extract_hierarchical_structure(self) -> List[DocumentSection]:
        """Extract document structure using font analysis"""
        sections = []
        current_section = None
        
        print("📚 Extracting document hierarchy...")
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        is_bold = span["flags"] & 2 ** 4
                        
                        if self._is_header(text, font_size, is_bold):
                            level = self._determine_header_level(font_size)
                            
                            if current_section:
                                current_section.page_end = page_num
                                sections.append(current_section)
                            
                            current_section = DocumentSection(
                                title=text,
                                level=level,
                                page_start=page_num,
                                page_end=None,
                                content="",
                                parent_section=self._find_parent_section(sections, level)
                            )
                        elif current_section:
                            current_section.content += text + " "
        
        if current_section:
            current_section.page_end = len(self.doc) - 1
            sections.append(current_section)
        
        self.sections = sections
        print(f"   ✓ Found {len(sections)} sections")
        return sections
    
    def _is_header(self, text: str, font_size: float, is_bold: bool) -> bool:
        """Heuristic to identify headers"""
        return (
            is_bold and 
            font_size > 11 and 
            len(text.split()) < 15 and
            (re.match(r'^\d+\.?\d*\s+', text) or text.isupper() or len(text.split()) < 6)
        )
    
    def _determine_header_level(self, font_size: float) -> int:
        """Determine hierarchy level from font size"""
        if font_size >= 16:
            return 1
        elif font_size >= 14:
            return 2
        else:
            return 3
    
    def _find_parent_section(self, sections: List[DocumentSection], level: int) -> Optional[str]:
        """Find parent section for hierarchy"""
        for section in reversed(sections):
            if section.level < level:
                return section.title
        return None
    
    def extract_formulas(self) -> List[ExtractedElement]:
        """Extract mathematical formulas"""
        print("🔢 Extracting formulas...")
        formulas = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            formula_patterns = [
                r'\$\$.*?\$\$',
                r'\$.*?\$',
                r'\\begin\{equation\}.*?\\end\{equation\}',
                r'\b[A-Z]\s*=\s*[^.]+',
            ]
            
            for pattern in formula_patterns:
                matches = re.finditer(pattern, text, re.DOTALL)
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    
                    formulas.append(ExtractedElement(
                        element_type='formula',
                        content=match.group(0),
                        page_number=page_num,
                        bbox=(0, 0, 0, 0),
                        caption=self._extract_caption(text, match.start()),
                        context=text[start:end]
                    ))
        
        print(f"   ✓ Found {len(formulas)} formulas")
        return formulas
    
    def extract_code_blocks(self) -> List[ExtractedElement]:
        """Extract code snippets"""
        print("💻 Extracting code blocks...")
        code_blocks = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                block_text = ""
                is_code = False
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        if "Courier" in span["font"] or "Mono" in span["font"]:
                            is_code = True
                        block_text += span["text"] + "\n"
                
                if is_code and len(block_text.strip()) > 20:
                    full_text = page.get_text()
                    code_start = full_text.find(block_text[:50])
                    
                    code_blocks.append(ExtractedElement(
                        element_type='code',
                        content=block_text.strip(),
                        page_number=page_num,
                        bbox=tuple(block["bbox"]),
                        caption=self._extract_caption(full_text, code_start),
                        context=full_text[max(0, code_start-200):min(len(full_text), code_start+len(block_text)+200)]
                    ))
        
        print(f"   ✓ Found {len(code_blocks)} code blocks")
        return code_blocks
    
    def extract_tables_and_figures(self) -> List[ExtractedElement]:
        """Extract tables and figures"""
        print("📊 Extracting tables and figures...")
        elements = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                
                if img_rects:
                    bbox = img_rects[0]
                    elements.append(ExtractedElement(
                        element_type='figure',
                        content=f"Figure {page_num}-{img_index}",
                        page_number=page_num,
                        bbox=tuple(bbox),
                        caption=self._find_figure_caption(page, bbox),
                        context=page.get_text()[:500]
                    ))
            
            # Extract tables
            tables = page.find_tables()
            for table_index, table in enumerate(tables):
                if table:
                    elements.append(ExtractedElement(
                        element_type='table',
                        content=table.extract(),
                        page_number=page_num,
                        bbox=tuple(table.bbox),
                        caption=self._find_table_caption(page, table.bbox),
                        context=""
                    ))
        
        print(f"   ✓ Found {len(elements)} tables/figures")
        return elements
    
    def _extract_caption(self, text: str, position: int) -> Optional[str]:
        """Extract caption near position"""
        window = text[max(0, position-200):min(len(text), position+200)]
        patterns = [
            r'Figure\s+\d+[.:]\s*([^\n]+)',
            r'Table\s+\d+[.:]\s*([^\n]+)',
            r'Equation\s+\d+[.:]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _find_figure_caption(self, page, bbox) -> Optional[str]:
        """Find figure caption"""
        search_rect = fitz.Rect(bbox.x0, bbox.y1, bbox.x1, bbox.y1 + 100)
        text = page.get_text("text", clip=search_rect)
        match = re.search(r'Figure\s+\d+[.:]\s*([^\n]+)', text, re.IGNORECASE)
        return match.group(0) if match else None
    
    def _find_table_caption(self, page, bbox) -> Optional[str]:
        """Find table caption"""
        search_rect = fitz.Rect(bbox[0], bbox[1] - 100, bbox[2], bbox[1])
        text = page.get_text("text", clip=search_rect)
        match = re.search(r'Table\s+\d+[.:]\s*([^\n]+)', text, re.IGNORECASE)
        return match.group(0) if match else None
    
    def process_document(self) -> Dict[str, Any]:
        """Complete processing pipeline using Markdown"""
        print("\n" + "="*70)
        print("🚀 STARTING DOCUMENT PROCESSING (MARKDOWN-BASED)")
        print("="*70 + "\n")
        
        # Convert to Markdown first
        self.convert_to_markdown()
        
        # Extract structure from original PDF
        self.extract_hierarchical_structure()
        
        # Extract multi-modal elements
        formulas = self.extract_formulas()
        code_blocks = self.extract_code_blocks()
        tables_figures = self.extract_tables_and_figures()
        self.elements = formulas + code_blocks + tables_figures
        
        # Chunk based on Markdown
        self.chunk_markdown()
        
        print("\n✅ Processing complete!")
        print(f"   - Markdown: {len(self.markdown_content)} chars")
        print(f"   - Sections: {len(self.sections)}")
        print(f"   - Elements: {len(self.elements)}")
        print(f"   - Chunks: {len(self.chunks)}")
        
        return {
            'markdown': self.markdown_content,
            'sections': [asdict(s) for s in self.sections],
            'elements': [asdict(e) for e in self.elements],
            'chunks': [asdict(c) for c in self.chunks]
        }
    
    def save_processed_data(self, output_dir: str = 'data/processed'):
        """Save all processed data including Markdown"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save Markdown
        with open(output_path / 'document.md', 'w', encoding='utf-8') as f:
            f.write(self.markdown_content)
        
        # Save JSON files
        with open(output_path / 'sections.json', 'w') as f:
            json.dump([asdict(s) for s in self.sections], f, indent=2)
        
        with open(output_path / 'elements.json', 'w') as f:
            json.dump([asdict(e) for e in self.elements], f, indent=2, default=str)
        
        with open(output_path / 'chunks.json', 'w') as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)
        
        print(f"💾 Data saved to {output_path}")
        print(f"   ✓ document.md - Full Markdown conversion")
        print(f"   ✓ sections.json - Document structure")
        print(f"   ✓ elements.json - Multi-modal elements")
        print(f"   ✓ chunks.json - Semantic chunks from Markdown")


# ============================================================================
# VECTOR STORES
# ============================================================================

class ChromaVectorStore:
    """ChromaDB-based vector store"""
    
    def __init__(self, collection_name: str = "aurelia_financial"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        self.client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Financial concepts from fintbx.pdf"}
        )
        
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Upload chunks"""
        print(f"🚀 Upserting {len(chunks)} chunks to ChromaDB...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk['chunk_id'] for chunk in batch]
            documents = [chunk['content'] for chunk in batch]
            embeddings = [self._get_embedding(chunk['content']) for chunk in batch]
            
            metadatas = [{
                'section_title': chunk['section_title'],
                'chunk_type': chunk['chunk_type'],
                'word_count': chunk['metadata']['word_count'],
            } for chunk in batch]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"   ✓ Batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        print(f"✅ Upload complete! Total: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'chunk_id': results['ids'][0][i],
                'score': 1 - results['distances'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return formatted


class PineconeVectorStore:
    """Pinecone-based vector store"""
    
    def __init__(self, index_name: str = "aurelia-financial"):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not installed. Run: pip install pinecone")
        
        api_key = os.getenv('PINECONE_API_KEY')
        self.pc = Pinecone(api_key=api_key)
        
        if index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=3072,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        self.index = self.pc.Index(index_name)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Upload chunks"""
        print(f"🚀 Upserting {len(chunks)} chunks to Pinecone...")
        
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk['content'])
            
            vectors.append({
                'id': chunk['chunk_id'],
                'values': embedding,
                'metadata': {
                    'content': chunk['content'][:1000],
                    'section_title': chunk['section_title'],
                    'chunk_type': chunk['chunk_type'],
                }
            })
            
            if len(vectors) >= batch_size or i == len(chunks) - 1:
                self.index.upsert(vectors=vectors)
                print(f"   ✓ Batch {i//batch_size + 1}")
                vectors = []
        
        print("✅ Upload complete!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search"""
        query_embedding = self._get_embedding(query)
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        return [{
            'chunk_id': match['id'],
            'score': match['score'],
            'metadata': match['metadata']
        } for match in results['matches']]


# ============================================================================
# HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """Combines dense + sparse retrieval"""
    
    def __init__(self, vector_store, chunks: List[Dict[str, Any]]):
        self.vector_store = vector_store
        self.chunks = chunks
        
        print("🔍 Building BM25 index...")
        corpus = [chunk['content'] for chunk in chunks]
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_lookup = {chunk['chunk_id']: chunk for chunk in chunks}
        print("✅ BM25 ready")
    
    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """Hybrid search with RRF"""
        # Dense retrieval
        dense_results = self.vector_store.search(query, top_k=top_k*2)
        
        # Sparse retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
        
        # Combine with RRF
        combined_scores = {}
        
        for rank, result in enumerate(dense_results):
            chunk_id = result['chunk_id']
            if chunk_id in self.chunk_lookup:
                combined_scores[chunk_id] = alpha * (1 / (60 + rank + 1))
        
        for rank, idx in enumerate(top_indices):
            if idx < len(self.chunks):
                chunk_id = self.chunks[idx]['chunk_id']
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + (1 - alpha) * (1 / (60 + rank + 1))
        
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Build results
        results = []
        for cid, score in ranked:
            if cid in self.chunk_lookup:
                results.append({
                    'chunk_id': cid,
                    'score': score,
                    'content': self.chunk_lookup[cid]['content'],
                    'metadata': self.chunk_lookup[cid]
                })
        
        return results
# ============================================================================
# MAIN PIPELINE
# ============================================================================

class AureliaLab1Pipeline:
    """Complete Lab 1 pipeline"""
    
    def __init__(self, pdf_path: str, vector_store_type: str = "chroma"):
        self.pdf_path = Path(pdf_path)
        self.vector_store_type = vector_store_type
        self.output_dir = Path("data/processed")
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print("🎯 AURELIA LAB 1 - COMPLETE PIPELINE (MARKDOWN-BASED)")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        # Step 1: Parse PDF and convert to Markdown
        print("STEP 1: PDF Parsing & Markdown Conversion")
        print("-"*70)
        parser = AdvancedPDFParser(str(self.pdf_path))
        parser.process_document()
        parser.save_processed_data(str(self.output_dir))
        
        # Step 2: Create vector store
        print("\n" + "="*70)
        print("STEP 2: Vector Store Creation")
        print("-"*70 + "\n")
        
        with open(self.output_dir / 'chunks.json') as f:
            chunks = json.load(f)
        
        if self.vector_store_type == "chroma":
            vector_store = ChromaVectorStore()
        else:
            vector_store = PineconeVectorStore()
        
        vector_store.upsert_chunks(chunks)
        
        # Step 3: Test retrieval
        print("\n" + "="*70)
        print("STEP 3: Testing Retrieval")
        print("-"*70 + "\n")
        
        retriever = HybridRetriever(vector_store, chunks)
        
        test_queries = [
            "What is the Sharpe Ratio?",
            "How to calculate duration?",
            "Black-Scholes model"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            results = retriever.search(query, top_k=3)
            
            for i, r in enumerate(results):
                print(f"  {i+1}. Score: {r['score']:.4f} | {r['metadata']['section_title']}")
                print(f"     {r['content'][:100]}...")
        
        # Step 4: Generate stats
        print("\n" + "="*70)
        print("STEP 4: Analysis")
        print("-"*70 + "\n")
        
        self._generate_stats(chunks)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n⏱️  Total time: {elapsed:.2f}s")
        print(f"✓ Processed {len(chunks)} chunks (Markdown-based)")
        print(f"✓ Vector store: {self.vector_store_type}")
        print(f"✓ Output: {self.output_dir}")
        print(f"\n📄 Generated files:")
        print(f"   - document.md (Full Markdown)")
        print(f"   - chunks.json (Semantic chunks)")
        print(f"   - sections.json (Structure)")
        print(f"   - elements.json (Multi-modal)")
    
    def _generate_stats(self, chunks):
        """Generate summary statistics"""
        lengths = [c['metadata']['word_count'] for c in chunks]
        types = Counter([c['chunk_type'] for c in chunks])
        
        # Check for Markdown-specific stats
        has_code = sum(1 for c in chunks if c['metadata'].get('has_code', False))
        has_formula = sum(1 for c in chunks if c['metadata'].get('has_formula', False))
        
        print("📊 Chunk Statistics (Markdown-based):")
        print(f"   Total: {len(chunks)}")
        print(f"   Avg length: {statistics.mean(lengths):.1f} words")
        print(f"   Median: {statistics.median(lengths):.1f} words")
        print(f"   With code blocks: {has_code}")
        print(f"   With formulas: {has_formula}")
        print(f"\n   Type distribution:")
        for ctype, count in types.most_common():
            pct = (count / len(chunks)) * 100
            print(f"     {ctype:15s}: {count:4d} ({pct:.1f}%)")


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AURELIA Lab 1 - Complete Pipeline (Markdown-Based)"
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        default='data/raw_pdfs/fintbx.pdf',
        help='Path to PDF file'
    )
    
    parser.add_argument(
        '--vector-store',
        type=str,
        choices=['chroma', 'pinecone'],
        default='chroma',
        help='Vector store type'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = AureliaLab1Pipeline(
        pdf_path=args.pdf,
        vector_store_type=args.vector_store
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
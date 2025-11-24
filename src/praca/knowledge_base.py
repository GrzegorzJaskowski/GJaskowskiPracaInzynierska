import os
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import fitz
import cv2
import pytesseract
from PIL import Image
import io
from dataclasses import dataclass
from datetime import datetime

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

PDF_PATH = os.getenv("PDF_PATH", str(Path(__file__).parent / "data" / "data.pdf"))
DB_PATH = os.getenv("DB_PATH", str(Path(__file__).parent.parent.parent / "faiss_db"))

DEFAULT_BATCH_SIZE = 50
DEFAULT_EMBEDDING_BATCH_SIZE = 100

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    page_number: int
    chunk_type: str  # 'text', 'table', 'image', 'figure'
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class PDFKnowledgeBase:
    """Main class for PDF content extraction and FAISS database management"""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 db_path: Optional[str] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE):
        """
        Initialize the knowledge base
        
        Args:
            model_name: Sentence transformer model for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            db_path: Path to store FAISS database
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if db_path is None:
            db_path = DB_PATH
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks = []
        self.metadata = []
        
        self.initialize_knowledge_base()
    
    def extract_pdf_content(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Extract all content from PDF using streaming approach
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        return self._extract_pdf_content_streaming(pdf_path)
    
    def _extract_pdf_content_streaming(self, pdf_path: str) -> List[DocumentChunk]:
        """Process PDF in batches to manage memory for large files"""
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            for start_page in range(0, total_pages, self.batch_size):
                end_page = min(start_page + self.batch_size, total_pages)
                
                batch_chunks = self._process_page_batch(doc, start_page, end_page)
                chunks.extend(batch_chunks)

                if len(chunks) >= self.embedding_batch_size:
                    self._process_embedding_batch(chunks[-self.embedding_batch_size:])
                    chunks = chunks[-self.embedding_batch_size//2:]
                
                gc.collect()
                
            
            doc.close()
            
        except Exception:
            return []
        
        return chunks
    
    def _process_page_batch(self, doc, start_page: int, end_page: int) -> List[DocumentChunk]:
        """Process a batch of pages"""
        chunks = []
        
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                
                text_chunks = self._extract_text_from_page(page, page_num + 1)
                chunks.extend(text_chunks)
                
                table_chunks = self._extract_tables_from_page(page, page_num + 1)
                chunks.extend(table_chunks)
                
                image_chunks = self._extract_images_from_page(page, page_num + 1)
                chunks.extend(image_chunks)
                
            except Exception:
                continue
        
        return chunks
    
    def _extract_text_from_page(self, page, page_number: int) -> List[DocumentChunk]:
        """Extract text content from a single page"""
        chunks = []
        
        try:
            text = page.get_text()
            
            if text.strip():
                text_chunks = self._split_text_into_chunks(text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        content=chunk_text,
                        page_number=page_number,
                        chunk_type="text",
                        metadata={
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            "extraction_method": "PyMuPDF4LLM"
                        }
                    )
                    chunks.append(chunk)
        
        except Exception:
            return []
        
        return chunks
    
    def _extract_tables_from_page(self, page, page_number: int) -> List[DocumentChunk]:
        """Extract table content from a single page using PyMuPDF4LLM"""
        chunks = []
        
        try:
            tables = page.find_tables()
            
            for table_id, table in enumerate(tables):
                try:
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 1:
                        table_text = self._format_table_as_text(table_data)
                        
                        try:
                            headers = table_data[0]
                            unique_headers = []
                            header_counts = {}
                            
                            for header in headers:
                                if header in header_counts:
                                    header_counts[header] += 1
                                    unique_headers.append(f"{header}_{header_counts[header]}")
                                else:
                                    header_counts[header] = 0
                                    unique_headers.append(header)
                            
                            table_df = pd.DataFrame(table_data[1:], columns=unique_headers)
                            table_json = table_df.to_json(orient='records', indent=2)
                        except Exception:
                            table_json = "{}"
                        
                        chunk = DocumentChunk(
                            content=f"Table {table_id + 1}:\n{table_text}",
                            page_number=page_number,
                            chunk_type="table",
                            metadata={
                                "table_index": table_id,
                                "table_json": table_json,
                                "extraction_method": "PyMuPDF4LLM",
                                "rows": len(table_data),
                                "columns": len(table_data[0]) if table_data else 0
                            }
                        )
                        chunks.append(chunk)
                
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return chunks
    
    def _extract_images_from_page(self, page, page_number: int) -> List[DocumentChunk]:
        """Extract images and perform selective OCR from a single page"""
        chunks = []
        
        try:
            image_list = page.get_images()
            
            for img_id, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        
                        if self._should_perform_ocr(pix.width, pix.height):
                            ocr_text = self._perform_ocr(img_data)
                            
                            if ocr_text.strip():
                                chunk = DocumentChunk(
                                    content=f"Image {img_id + 1} (OCR):\n{ocr_text}",
                                    page_number=page_number,
                                    chunk_type="image",
                                    metadata={
                                        "image_index": img_id,
                                        "extraction_method": "PyMuPDF4LLM_OCR",
                                        "image_format": "png",
                                        "image_size": f"{pix.width}x{pix.height}"
                                    }
                                )
                                chunks.append(chunk)
                        else:
                            chunk = DocumentChunk(
                                content=f"Image {img_id + 1} (no OCR - too large/decorative)",
                                page_number=page_number,
                                chunk_type="image",
                                metadata={
                                    "image_index": img_id,
                                    "extraction_method": "PyMuPDF4LLM_skip_OCR",
                                    "image_format": "png",
                                    "image_size": f"{pix.width}x{pix.height}",
                                    "ocr_skipped": True
                                }
                            )
                            chunks.append(chunk)
                    
                    pix = None
                    
                except Exception:
                    continue
        
        except Exception:
            return []
        
        return chunks
    
    def _should_perform_ocr(self, width: int, height: int) -> bool:
        """Determine if OCR is worth performing on this image"""
        if width < 100 or height < 100:
            return False
        if width > 2000 or height > 2000:
            return False
        
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 10:
            return False
        
        return True
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
        
        return chunks
    
    def _format_table_as_text(self, table: List[List[str]]) -> str:
        """Format table as readable text"""
        if not table:
            return ""
        
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        lines = []
        for row in cleaned_table:
            lines.append(" | ".join(row))
        
        return "\n".join(lines)
    
    def _perform_ocr(self, image_data: bytes) -> str:
        """Perform OCR on image data"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(processed, config='--psm 6')
            
            return text.strip()
            
        except Exception:
            return ""
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Create embeddings for document chunks"""
        return self.create_embeddings_batched(chunks, self.embedding_batch_size)
    
    def create_embeddings_batched(self, chunks: List[DocumentChunk], batch_size: int = None) -> List[DocumentChunk]:
        """Create embeddings in smaller batches"""
        if batch_size is None:
            batch_size = self.embedding_batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                embeddings = self.embedder.encode(texts, show_progress_bar=True)
                
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                
                del texts, embeddings
                gc.collect()
                
            except Exception:
                continue
        
        return chunks
    
    def _process_embedding_batch(self, chunks: List[DocumentChunk]):
        """Process a batch of chunks for embeddings and store in FAISS"""
        if not chunks:
            return
        
        self.create_embeddings_batched(chunks, len(chunks))
        
        self.chunks.extend(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        embeddings_matrix = np.array([chunk.embedding for chunk in chunks if chunk.embedding is not None])
        if len(embeddings_matrix) > 0:
            self.index.add(embeddings_matrix.astype('float32'))
        
        for chunk in chunks:
            self.metadata.append({
                'content': chunk.content,
                'page_number': chunk.page_number,
                'chunk_type': chunk.chunk_type,
                'metadata': chunk.metadata
            })
        
        del embeddings_matrix
        gc.collect()
    
    def build_database(self, pdf_path: str):
        """Build FAISS database from PDF content"""
        
        chunks = self.extract_pdf_content(pdf_path)
        
        if not chunks:
            return
        
        chunks = self.create_embeddings(chunks)
        
        self.chunks.extend(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        embeddings_matrix = np.array([chunk.embedding for chunk in chunks if chunk.embedding is not None])
        if len(embeddings_matrix) > 0:
            self.index.add(embeddings_matrix.astype('float32'))
        
        for chunk in chunks:
            self.metadata.append({
                'content': chunk.content,
                'page_number': chunk.page_number,
                'chunk_type': chunk.chunk_type,
                'metadata': chunk.metadata
            })
        
        self._save_database()
        
        del embeddings_matrix
        gc.collect()

        stats =self.get_database_stats()
        print(stats)
        
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the database for relevant content
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.embedder.encode([query])
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, id in zip(scores[0], indices[0]):
            if id < len(self.chunks):
                chunk = self.chunks[id]
                result = {
                    'content': chunk.content,
                    'page_number': chunk.page_number,
                    'chunk_type': chunk.chunk_type,
                    'metadata': chunk.metadata,
                    'similarity_score': float(score)
                }
                results.append(result)
        
        return results
    
    def _save_database(self):
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.db_path / "faiss_index.bin"))
            
            with open(self.db_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        except Exception:
            pass
    
    def _load_database(self):
        """Load existing FAISS index and metadata"""
        try:
            index_path = self.db_path / "faiss_index.bin"
            metadata_path = self.db_path / "metadata.json"
            
            if index_path.exists() and metadata_path.exists():
                self.index = faiss.read_index(str(index_path))
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                self.chunks = []
                for meta in self.metadata:
                    chunk = DocumentChunk(
                        content=meta['content'],
                        page_number=meta['page_number'],
                        chunk_type=meta['chunk_type'],
                        metadata=meta['metadata']
                    )
                    self.chunks.append(chunk)
                
            
        except Exception:
            pass
    
    def database_exists(self) -> bool:
        """Check if the database already exists on disk"""
        database_path = self.db_path / "faiss_index.bin"
        metadata_path = self.db_path / "metadata.json"
        return database_path.exists() and metadata_path.exists()
    
    def rebuild_database(self, pdf_path: str = None):
        """Force rebuild the database from PDF"""
        if pdf_path is None:
            pdf_path = PDF_PATH
        
        if not os.path.exists(pdf_path):
            return False
        
        self.build_database(pdf_path)
        return True
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {
            'total_chunks': len(self.chunks),
            'chunk_types': {},
            'pages_covered': set(),
            'total_content_length': 0
        }
        
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            
            stats['pages_covered'].add(chunk.page_number)
            
            stats['total_content_length'] += len(chunk.content)
        
        stats['pages_covered'] = sorted(list(stats['pages_covered']))
        stats['unique_pages'] = len(stats['pages_covered'])
        
        return stats

    def initialize_knowledge_base(self):
        """Initialize the knowledge base"""
        pdf_path = PDF_PATH
        
        if not pdf_path or not os.path.exists(pdf_path):
            return
        
        if self.database_exists():
            self._load_database()
        else:
            self.build_database(pdf_path)

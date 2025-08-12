"""
RAG Tool for Local Document Search

Implements a local Retrieval-Augmented Generation tool that allows the agent
to search through local documents and provide contextual answers.

This tool follows the existing ToolInterface pattern and integrates seamlessly
with the LangGraph agent workflow.
"""

import os
import time
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import PyPDF2
except ImportError as e:
    # Handle missing dependencies gracefully
    missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
    raise ImportError(
        f"RAG dependencies not installed. Missing: {missing_dep}. "
        "Run: pip install sentence-transformers numpy scikit-learn PyPDF2"
    )

from core.interfaces.agent_service import ToolInterface
from core.models.agent import ToolInfo
from config.constants import RAG_CONFIG
from utils.logger import get_logger, log_performance, log_tool_execution


class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    def __init__(self, content: str, source_file: str, chunk_index: int, 
                 start_char: int = 0, end_char: int = 0):
        self.content = content
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.embedding: Optional[np.ndarray] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char
        }


class RAGTool(ToolInterface):
    """Local RAG tool for document search and retrieval."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = RAG_CONFIG
        
        # Tool metadata
        self.name = "rag_search"
        self.description = "Search through local documents and knowledge base. Use this when the user asks about information that might be in your local documentation, procedures, or knowledge base files."
        
        # RAG components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.document_chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Cache settings
        self.embeddings_cache_file = self.config.get("embeddings_cache_file", "./cache/rag_embeddings.pkl")
        self.cache_enabled = self.config.get("cache_embeddings", True)
        
        # Ensure cache directory exists
        cache_dir = Path(self.embeddings_cache_file).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Document tracking
        self.indexed_files: Dict[str, float] = {}  # file_path -> last_modified_time
        self.last_refresh_time = 0
        
        # Initialize the tool
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG tool by loading or creating embeddings."""
        try:
            self.logger.info("Initializing RAG tool", config=self.config)
            
            if not self.config.get("enabled", True):
                self.logger.info("RAG tool disabled in configuration")
                return
            
            # Load embedding model
            self._load_embedding_model()
            
            # Load or create document index
            self._load_or_create_index()
            
            self.logger.info("RAG tool initialized successfully", 
                           document_count=len(self.document_chunks),
                           embedding_model=self.config.get("embedding_model"))
        
        except Exception as e:
            self.logger.error("Failed to initialize RAG tool", error=str(e))
            # Don't raise exception - tool should gracefully degrade
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.logger.info("Loading embedding model", model=model_name)
            
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.error("Failed to load embedding model", error=str(e))
            raise
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if self.cache_enabled and os.path.exists(self.embeddings_cache_file):
            try:
                self._load_cached_embeddings()
                
                # Check if refresh is needed
                if self._should_refresh_index():
                    self.logger.info("Index refresh needed, rebuilding...")
                    self._build_document_index()
                else:
                    self.logger.info("Using cached embeddings", 
                                   document_count=len(self.document_chunks))
            except Exception as e:
                self.logger.warning("Failed to load cached embeddings, rebuilding", error=str(e))
                self._build_document_index()
        else:
            self.logger.info("Building new document index...")
            self._build_document_index()
    
    def _should_refresh_index(self) -> bool:
        """Check if the document index should be refreshed."""
        if not self.config.get("auto_refresh", True):
            return False
        
        # Check time-based refresh
        refresh_interval = self.config.get("refresh_interval", 3600)
        if time.time() - self.last_refresh_time > refresh_interval:
            return True
        
        # Check for file changes
        documents_path = Path(self.config.get("documents_path", "./documents"))
        if not documents_path.exists():
            return False
        
        for file_path in self._get_document_files(documents_path):
            file_stat = file_path.stat()
            stored_mtime = self.indexed_files.get(str(file_path), 0)
            
            if file_stat.st_mtime > stored_mtime:
                self.logger.info("File change detected", file=str(file_path))
                return True
        
        return False
    
    def _build_document_index(self):
        """Build the document index from scratch."""
        start_time = time.time()
        
        try:
            # Clear existing data
            self.document_chunks = []
            self.indexed_files = {}
            
            # Scan documents
            documents_path = Path(self.config.get("documents_path", "./documents"))
            if not documents_path.exists():
                self.logger.warning("Documents path does not exist", path=str(documents_path))
                return
            
            # Process all documents
            for file_path in self._get_document_files(documents_path):
                try:
                    self._process_document(file_path)
                    self.indexed_files[str(file_path)] = file_path.stat().st_mtime
                except Exception as e:
                    self.logger.error("Failed to process document", file=str(file_path), error=str(e))
            
            # Create embeddings
            if self.document_chunks and self.embedding_model:
                self._create_embeddings()
            
            # Cache the results
            if self.cache_enabled:
                self._save_cached_embeddings()
            
            self.last_refresh_time = time.time()
            
            duration = time.time() - start_time
            self.logger.info("Document index built successfully", 
                           document_count=len(self.document_chunks),
                           duration_seconds=round(duration, 2))
        
        except Exception as e:
            self.logger.error("Failed to build document index", error=str(e))
    
    def _get_document_files(self, documents_path: Path) -> List[Path]:
        """Get all supported document files."""
        supported_extensions = self.config.get("supported_extensions", [".md", ".txt", ".pdf"])
        files = []
        
        for ext in supported_extensions:
            files.extend(documents_path.rglob(f"*{ext}"))
        
        return sorted(files)
    
    def _process_document(self, file_path: Path):
        """Process a single document into chunks."""
        try:
            content = self._extract_text(file_path)
            if not content or not content.strip():
                self.logger.warning("Empty document content", file=str(file_path))
                return
            
            chunks = self._create_chunks(content, str(file_path))
            self.document_chunks.extend(chunks)
            
            self.logger.debug("Processed document", 
                            file=str(file_path), 
                            chunks_created=len(chunks),
                            content_length=len(content))
        
        except Exception as e:
            self.logger.error("Failed to process document", file=str(file_path), error=str(e))
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from a file."""
        try:
            if file_path.suffix.lower() == ".pdf":
                return self._extract_pdf_text(file_path)
            else:
                # Handle text files (md, txt, etc.)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            self.logger.error("Failed to extract text", file=str(file_path), error=str(e))
            return ""
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        except Exception as e:
            self.logger.error("Failed to extract PDF text", file=str(file_path), error=str(e))
            return ""
    
    def _create_chunks(self, content: str, source_file: str) -> List[DocumentChunk]:
        """Create chunks from document content."""
        chunk_size = self.config.get("chunk_size", 500)
        chunk_overlap = self.config.get("chunk_overlap", 50)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space within reasonable distance
                break_point = content.rfind(' ', start, end)
                if break_point > start + chunk_size * 0.8:  # Don't break too early
                    end = break_point
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    content=chunk_content,
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
    
    def _create_embeddings(self):
        """Create embeddings for all document chunks."""
        try:
            self.logger.info("Creating embeddings for document chunks", 
                           chunk_count=len(self.document_chunks))
            
            texts = [chunk.content for chunk in self.document_chunks]
            
            # Create embeddings in batches to manage memory
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            self.embeddings = np.vstack(all_embeddings)
            
            # Store embeddings in chunks for reference
            for i, chunk in enumerate(self.document_chunks):
                chunk.embedding = self.embeddings[i]
            
            self.logger.info("Embeddings created successfully", 
                           embedding_shape=self.embeddings.shape)
        
        except Exception as e:
            self.logger.error("Failed to create embeddings", error=str(e))
            raise
    
    def _save_cached_embeddings(self):
        """Save embeddings and metadata to cache file."""
        try:
            cache_data = {
                "document_chunks": [chunk.to_dict() for chunk in self.document_chunks],
                "embeddings": self.embeddings,
                "indexed_files": self.indexed_files,
                "last_refresh_time": self.last_refresh_time,
                "config_hash": self._get_config_hash()
            }
            
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.info("Embeddings cached successfully", cache_file=self.embeddings_cache_file)
        
        except Exception as e:
            self.logger.error("Failed to save cached embeddings", error=str(e))
    
    def _load_cached_embeddings(self):
        """Load embeddings and metadata from cache file."""
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache consistency
            if cache_data.get("config_hash") != self._get_config_hash():
                raise ValueError("Configuration has changed, cache invalid")
            
            # Restore document chunks
            self.document_chunks = []
            for chunk_data in cache_data.get("document_chunks", []):
                chunk = DocumentChunk(
                    content=chunk_data["content"],
                    source_file=chunk_data["source_file"],
                    chunk_index=chunk_data["chunk_index"],
                    start_char=chunk_data.get("start_char", 0),
                    end_char=chunk_data.get("end_char", 0)
                )
                self.document_chunks.append(chunk)
            
            # Restore embeddings
            self.embeddings = cache_data.get("embeddings")
            
            # Restore metadata
            self.indexed_files = cache_data.get("indexed_files", {})
            self.last_refresh_time = cache_data.get("last_refresh_time", 0)
            
            # Link embeddings to chunks
            if self.embeddings is not None and len(self.embeddings) == len(self.document_chunks):
                for i, chunk in enumerate(self.document_chunks):
                    chunk.embedding = self.embeddings[i]
            
            self.logger.info("Cached embeddings loaded successfully", 
                           document_count=len(self.document_chunks))
        
        except Exception as e:
            self.logger.error("Failed to load cached embeddings", error=str(e))
            raise
    
    def _get_config_hash(self) -> str:
        """Get hash of relevant configuration for cache validation."""
        config_str = f"{self.config.get('chunk_size', 500)}-{self.config.get('embedding_model', '')}-{self.config.get('documents_path', '')}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    @log_performance("rag_search")
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG search on local documents.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters
                - max_results: Maximum number of results (default from config)
                - similarity_threshold: Minimum similarity score (default from config)
        
        Returns:
            Dict containing search results and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info("Executing RAG search", query=query[:100])
            
            # Check if RAG is properly initialized
            if not self.embedding_model or not self.document_chunks or self.embeddings is None:
                return {
                    "success": False,
                    "error": "RAG tool not properly initialized",
                    "results": "",
                    "metadata": {
                        "tool": "rag_search",
                        "query": query,
                        "status": "not_initialized"
                    }
                }
            
            # Get parameters
            max_results = kwargs.get("max_results", self.config.get("max_results", 3))
            similarity_threshold = kwargs.get("similarity_threshold", self.config.get("similarity_threshold", 0.5))  # Lower threshold for better recall
            
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top results above threshold
            result_indices = []
            for i, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    result_indices.append((i, similarity))
            
            # Sort by similarity and take top results
            result_indices.sort(key=lambda x: x[1], reverse=True)
            result_indices = result_indices[:max_results]
            
            if not result_indices:
                # No results above threshold - try with lower threshold as fallback
                fallback_threshold = 0.3
                fallback_indices = []
                for i, similarity in enumerate(similarities):
                    if similarity >= fallback_threshold:
                        fallback_indices.append((i, similarity))
                
                fallback_indices.sort(key=lambda x: x[1], reverse=True)
                fallback_indices = fallback_indices[:max_results]
                
                duration_ms = round((time.time() - start_time) * 1000)
                
                if fallback_indices:
                    # Found results with lower threshold
                    results_text = f"Found {len(fallback_indices)} potentially relevant documents (lower confidence matches):\n\n"
                    results_text += self._format_search_results(query, fallback_indices)
                    
                    result = {
                        "success": True,
                        "results": results_text,
                        "metadata": {
                            "tool": "rag_search",
                            "query": query,
                            "documents_searched": len(self.document_chunks),
                            "results_found": len(fallback_indices),
                            "total_results": len(fallback_indices),  # For UI compatibility
                            "max_similarity": float(np.max(similarities)) if len(similarities) > 0 else 0,
                            "similarity_threshold": similarity_threshold,
                            "fallback_threshold": fallback_threshold,
                            "using_fallback": True,
                            "search_time_ms": duration_ms,
                            "source_files": list(set(self.document_chunks[idx].source_file for idx, _ in fallback_indices)),
                            "provider": "RAG Search (Fallback)"  # For UI consistency
                        }
                    }
                else:
                    # No results even with fallback
                    max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0
                    result = {
                        "success": True,
                        "results": f"No relevant documents found in the local knowledge base.\n\nSearched {len(self.document_chunks)} document chunks. Highest similarity: {max_sim:.3f} (threshold: {similarity_threshold})\n\nTip: Try rephrasing your query or check if relevant documents are in the {self.config.get('documents_path', './documents')} folder.",
                        "metadata": {
                            "tool": "rag_search",
                            "query": query,
                            "documents_searched": len(self.document_chunks),
                            "results_found": 0,
                            "total_results": 0,  # For UI compatibility
                            "max_similarity": max_sim,
                            "similarity_threshold": similarity_threshold,
                            "search_time_ms": duration_ms,
                            "documents_path": self.config.get("documents_path", "./documents"),
                            "provider": "RAG Search"  # For UI consistency
                        }
                    }
                
                log_tool_execution(
                    tool_name="rag_search",
                    action="search",
                    query=query,
                    result=result,
                    duration_ms=duration_ms,
                    success=True,
                    results_count=0
                )
                
                return result
            
            # Format results
            results_text = self._format_search_results(query, result_indices)
            
            duration_ms = round((time.time() - start_time) * 1000)
            
            # Get unique source files
            source_files = list(set(
                self.document_chunks[idx].source_file for idx, _ in result_indices
            ))
            
            result = {
                "success": True,
                "results": results_text,
                "metadata": {
                    "tool": "rag_search",
                    "query": query,
                    "documents_searched": len(self.document_chunks),
                    "results_found": len(result_indices),
                    "total_results": len(result_indices),  # For UI compatibility
                    "source_files": source_files,
                    "similarity_scores": [float(score) for _, score in result_indices],
                    "similarity_threshold": similarity_threshold,
                    "search_time_ms": duration_ms,
                    "provider": "RAG Search"  # For UI consistency
                }
            }
            
            log_tool_execution(
                tool_name="rag_search",
                action="search",
                query=query,
                result=result,
                duration_ms=duration_ms,
                success=True,
                results_count=len(result_indices),
                source_files=len(source_files)
            )
            
            return result
        
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000)
            
            result = {
                "success": False,
                "error": f"RAG search failed: {str(e)}",
                "results": "",
                "metadata": {
                    "tool": "rag_search",
                    "query": query,
                    "error": str(e),
                    "search_time_ms": duration_ms,
                    "results_found": 0,
                    "total_results": 0,  # For UI compatibility
                    "provider": "RAG Search"  # For UI consistency
                }
            }
            
            log_tool_execution(
                tool_name="rag_search",
                action="search",
                query=query,
                result=result,
                duration_ms=duration_ms,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            self.logger.error("RAG search failed", query=query, error=str(e))
            return result
    
    def _format_search_results(self, query: str, result_indices: List[Tuple[int, float]]) -> str:
        """Format search results into a readable response."""
        results_parts = []
        results_parts.append(f"Found {len(result_indices)} relevant documents for your query:")
        results_parts.append("")
        
        for i, (chunk_idx, similarity) in enumerate(result_indices, 1):
            chunk = self.document_chunks[chunk_idx]
            
            # Get just the filename for cleaner display
            source_file = Path(chunk.source_file).name
            
            results_parts.append(f"**Result {i}** (from {source_file}, similarity: {similarity:.3f}):")
            results_parts.append(chunk.content)
            results_parts.append("")
        
        return "\n".join(results_parts)
    
    def get_tool_info(self) -> ToolInfo:
        """Get tool information for the agent."""
        return ToolInfo(
            name=self.name,
            description=self.description,
            parameters={
                "max_results": "Maximum number of results to return",
                "similarity_threshold": "Minimum similarity score for results"
            },
            metadata={
                "type": "rag_search",
                "documents_indexed": len(self.document_chunks),
                "embedding_model": self.config.get("embedding_model"),
                "documents_path": self.config.get("documents_path")
            }
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the RAG tool."""
        return {
            "enabled": self.config.get("enabled", True),
            "initialized": self.embedding_model is not None,
            "documents_indexed": len(self.document_chunks),
            "documents_path": self.config.get("documents_path"),
            "embedding_model": self.config.get("embedding_model"),
            "cache_enabled": self.cache_enabled,
            "last_refresh": self.last_refresh_time,
            "indexed_files": len(self.indexed_files)
        }
    
    def refresh_index(self) -> Dict[str, Any]:
        """Manually refresh the document index."""
        try:
            self.logger.info("Manual refresh of RAG index requested")
            self._build_document_index()
            
            return {
                "success": True,
                "message": "Index refreshed successfully",
                "documents_indexed": len(self.document_chunks),
                "last_refresh": self.last_refresh_time
            }
        
        except Exception as e:
            self.logger.error("Failed to refresh RAG index", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to refresh index"
            }
    
    def debug_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Debug method to show similarity scores for troubleshooting."""
        try:
            if not self.embedding_model or not self.document_chunks or self.embeddings is None:
                return {
                    "error": "RAG tool not initialized",
                    "initialized": False,
                    "document_count": len(self.document_chunks),
                    "has_embeddings": self.embeddings is not None,
                    "has_model": self.embedding_model is not None
                }
            
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top K similarities regardless of threshold
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            debug_results = []
            for idx in top_indices:
                chunk = self.document_chunks[idx]
                debug_results.append({
                    "similarity": float(similarities[idx]),
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "source_file": Path(chunk.source_file).name,
                    "chunk_index": chunk.chunk_index
                })
            
            return {
                "query": query,
                "documents_indexed": len(self.document_chunks),
                "max_similarity": float(np.max(similarities)),
                "min_similarity": float(np.min(similarities)),
                "avg_similarity": float(np.mean(similarities)),
                "threshold": self.config.get("similarity_threshold", 0.5),
                "top_results": debug_results,
                "documents_path": self.config.get("documents_path", "./documents")
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }

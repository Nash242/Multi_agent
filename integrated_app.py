from typing import Optional
from integrated_workflow import build_integrated_workflow
from helper import cleanup_clients
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, EMBEDDING_MODEL, DEFAULT_K

class IntegratedApp:
    """Integrated RAG + Weather application."""
    
    def __init__(self):
        self.workflow = build_integrated_workflow()
    
    def query(
        self,
        question: str,
        pdf_path: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embed_model: str = EMBEDDING_MODEL,
        k: int = DEFAULT_K,
        force_rebuild: bool = False
    ) -> dict:
        """
        Query the integrated system.
        
        Args:
            question: User's question
            pdf_path: Optional path to PDF for RAG queries
            chunk_size: Chunk size for text splitting (RAG)
            chunk_overlap: Chunk overlap (RAG)
            embed_model: Embedding model name (RAG)
            k: Number of documents to retrieve (RAG)
            force_rebuild: Force rebuild index (RAG)
            
        Returns:
            Result dictionary with answer and metadata
        """
        input_state = {
            "question": question,
            "pdf_path": pdf_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embed_model": embed_model,
            "k": k,
            "force_rebuild": force_rebuild
        }
        
        result = self.workflow.invoke(input_state)
        return result
    
    def get_sources(self, result: dict) -> list:
        """Extract source information from retrieved documents."""
        sources = []
        for doc in result.get("retrieved_docs", []):
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        return sources
    
    def cleanup(self):
        """Cleanup resources."""
        cleanup_clients()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


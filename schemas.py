import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_core.documents import Document

class RAGState(TypedDict, total=False):
    """State schema for the RAG workflow."""
    pdf_path: str
    question: str
    chunk_size: int
    chunk_overlap: int
    embed_model: str
    force_rebuild: bool
    
    # Intermediate data
    docs: List[Document]
    splits: List[Document]
    collection_name: str
    persist_dir: str
    cache_hit: bool
    vectorstore: object  # Qdrant vectorstore
    retrieved_docs: List[Document]
    context: str
    answer: str
    
    # Metadata
    steps: Annotated[List[str], operator.add]


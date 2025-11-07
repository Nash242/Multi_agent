import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.documents import Document

class UnifiedState(TypedDict, total=False):
    """Unified state for both RAG and Weather agents."""
    # Input
    question: str
    pdf_path: Optional[str]
    
    # Routing
    agent_type: Literal["rag", "weather", "unknown"]
    
    # RAG specific
    chunk_size: int
    chunk_overlap: int
    embed_model: str
    force_rebuild: bool
    k: int
    docs: List[Document]
    splits: List[Document]
    collection_name: str
    persist_dir: str
    cache_hit: bool
    vectorstore: object
    retrieved_docs: List[Document]
    context: str
    
    # Weather specific
    city: Optional[str]
    state: Optional[str]
    weather_data: Optional[dict]
    
    # Output
    answer: str
    metadata: dict
    
    # Tracking
    steps: Annotated[List[str], operator.add]


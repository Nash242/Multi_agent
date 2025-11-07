import json
import hashlib
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import INDEX_ROOT, LLM_MODEL

# Global caches
_client_cache = {}
_vectorstore_cache = {}
_pdf_context_cache = {}  


def get_or_create_client(persist_dir: str) -> QdrantClient:
    """Get or create a cached Qdrant client for the given directory."""
    if persist_dir not in _client_cache:
        _client_cache[persist_dir] = QdrantClient(path=persist_dir)
    return _client_cache[persist_dir]

def cleanup_clients():
    """Cleanup all cached Qdrant clients."""
    global _client_cache, _vectorstore_cache
    for client in _client_cache.values():
        try:
            client.close()
        except Exception:
            pass
    _client_cache.clear()
    _vectorstore_cache.clear()

@traceable(name="load_pdf", tags=["loader"])
def load_pdf(path: str) -> List[Document]:
    """Load PDF and return list of documents."""
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="split_documents", tags=["splitter"])
def split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def extract_pdf_context(pdf_path: str, model_name: str = LLM_MODEL) -> str:
    """
    Extract a comprehensive summary/context from a PDF using existing components.
    Uses caching to avoid re-summarizing the same PDF.
    
    Args:
        pdf_path: Path to PDF file
        model_name: LLM model for summarization
        
    Returns:
        Summarized context string
    """
    # Check cache first
    cache_key = f"{pdf_path}:{model_name}"
    if cache_key in _pdf_context_cache:
        print(f"[PDF CONTEXT] Using cached summary for {pdf_path}")
        return _pdf_context_cache[cache_key]
    
    print(f"[PDF CONTEXT] Generating summary for {pdf_path}...")
    
    # Step 1: Load PDF using existing function
    docs = load_pdf(pdf_path)
    
    # Step 2: Split into larger chunks for summarization
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    # Step 3: Initialize LLM and summarization chain
    model = ChatOpenAI(model=model_name, temperature=0)
    
    chunk_prompt = ChatPromptTemplate.from_template("""
You are an expert document analyzer.
Summarize the following document section concisely but thoroughly.
Focus on **main topics, key concepts, and important details**.

Document Section:
{content}

Provide a structured summary covering:
- Main Topics
- Key Concepts  
- Important Details
- Purpose/Context

Keep it concise but informative (150-200 words).
""")
    
    summarizer_chain = chunk_prompt | model | StrOutputParser()
    
    # Step 4: Summarize chunks (limit to first 10 for performance)
    all_summaries = []
    max_chunks = min(10, len(chunks))
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        text = chunk.page_content.strip()
        if not text or len(text) < 100:
            continue
        
        try:
            print(f"[PDF CONTEXT] Processing chunk {i+1}/{max_chunks}...")
            summary = summarizer_chain.invoke({"content": text})
            all_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Summarization error on chunk {i+1}: {e}")
            continue
    
    if not all_summaries:
        return "Unable to extract meaningful context from the PDF."
    
    # Step 5: Combine into final comprehensive summary
    combined_text = "\n\n".join(all_summaries)
    
    final_prompt = ChatPromptTemplate.from_template("""
You are creating a comprehensive overview of a PDF document.

Combine the following partial summaries into a single, cohesive document overview.
Keep it concise (300-400 words) but information-rich and well-structured.

Partial Summaries:
{summaries}

Create a final overview with these sections:
- **Document Title/Subject**: What is this document about?
- **Main Themes**: Core topics covered
- **Key Concepts**: Important ideas, definitions, or frameworks
- **Purpose/Use Cases**: What is this document for? Who would use it?

Maintain a professional, structured tone.
""")
    
    final_chain = final_prompt | model | StrOutputParser()
    
    try:
        print(f"[PDF CONTEXT] Creating final summary...")
        final_summary = final_chain.invoke({"summaries": combined_text})
        
        # Cache the result
        _pdf_context_cache[cache_key] = final_summary.strip()
        print(f"[PDF CONTEXT] ✓ Summary generated and cached")
        
        return final_summary.strip()
    except Exception as e:
        print(f"⚠️ Final summarization error: {e}")
        return combined_text[:1000]  # Fallback to truncated combined summaries


@traceable(name="extract_all_pdf_contexts", tags=["summarization"])
def extract_all_pdf_contexts(folder_path: str, model_name: str = LLM_MODEL) -> str:
    """
    Summarize all PDFs in a given folder and return a combined context.

    Args:
        folder_path (str): Path to the folder containing PDFs.

    Returns:
        str: Combined context summary of all PDFs.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {folder}")

    all_summaries: List[str] = []
    for pdf in pdf_files:
        summary = extract_pdf_context(str(pdf), model_name)
        all_summaries.append(f"📄 {pdf.name}\n{summary}\n")

    # Combine all summaries into one context
    model = ChatOpenAI(model=model_name, temperature=0)
    merge_prompt = ChatPromptTemplate.from_template("""
You are creating a unified context summary for multiple PDF documents.
Combine all given document summaries into one cohesive overview.
Keep the structure clear and concise.

PDF Summaries:
{summaries}
""")
    merge_chain = merge_prompt | model | StrOutputParser()
    final_context = merge_chain.invoke({"summaries": "\n".join(all_summaries)})

    return final_context.strip()

@traceable(name="build_qdrant_vectorstore", tags=["vectorstore", "build"])
def build_qdrant_vectorstore(
    splits: List[Document],
    collection_name: str,
    persist_dir: str,
    embed_model_name: str
):
    """Build persistent Qdrant vectorstore from document splits."""
    embeddings = OpenAIEmbeddings(model=embed_model_name)
    
    # Get or create cached client
    client = get_or_create_client(persist_dir)
    
    # Get embedding dimension
    test_embedding = embeddings.embed_query("test")
    vector_size = len(test_embedding)
    
    # Delete collection if exists
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # Create vectorstore with cached client
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    # Add documents
    vectorstore.add_documents(splits)
    
    return vectorstore

@traceable(name="load_qdrant_vectorstore", tags=["vectorstore", "load"])
def load_qdrant_vectorstore(
    collection_name: str,
    persist_dir: str,
    embed_model_name: str
):
    """Load existing Qdrant vectorstore from disk with caching."""
    cache_key = f"{persist_dir}:{collection_name}:{embed_model_name}"
    
    # Return cached vectorstore if available
    if cache_key in _vectorstore_cache:
        return _vectorstore_cache[cache_key]
    
    embeddings = OpenAIEmbeddings(model=embed_model_name)
    
    # Get cached client
    client = get_or_create_client(persist_dir)
    
    # Load vectorstore
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    # Cache it
    _vectorstore_cache[cache_key] = vectorstore
    
    return vectorstore

def save_metadata(persist_dir: Path, metadata: dict):
    """Save metadata to disk."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    meta_path = persist_dir / "meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

def load_metadata(persist_dir: Path) -> Optional[dict]:
    """Load metadata from disk."""
    meta_path = persist_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None

def file_fingerprint(path: str) -> dict:
    """Generate fingerprint of a file based on content hash and metadata."""
    p = Path(path)
    
    # Quick check: use mtime and size for fast comparison
    stat = p.stat()
    quick_key = f"{stat.st_size}_{int(stat.st_mtime)}"
    
    # Cache the hash calculation
    cache_file = Path(f".cache_{hashlib.md5(path.encode()).hexdigest()}.json")
    
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("quick_key") == quick_key:
                return cached
        except Exception:
            pass
    
    # Calculate hash if needed
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    
    result = {
        "sha256": h.hexdigest(),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "quick_key": quick_key
    }
    
    # Cache the result
    try:
        cache_file.write_text(json.dumps(result))
    except Exception:
        pass
    
    return result

def generate_collection_name(pdf_path: str) -> str:
    """Generate collection name based only on PDF content hash."""
    fingerprint = file_fingerprint(pdf_path)
    # Use only PDF hash for collection name
    return f"pdf_{fingerprint['sha256'][:24]}"

def get_index_metadata(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model: str) -> dict:
    """Generate metadata for index validation."""
    return {
        "pdf_fingerprint": file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model,
        "format": "v2",
    }

def is_index_valid(collection_name: str, persist_dir: str, 
                   pdf_path: str, chunk_size: int, 
                   chunk_overlap: int, embed_model: str) -> bool:
    """Check if existing index is valid for current parameters."""
    persist_path = Path(persist_dir)
    
    # Check if directory exists
    if not persist_path.exists():
        return False
    
    # Check if collection exists
    try:
        client = get_or_create_client(persist_dir)
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            return False
    except Exception:
        return False
    
    # Load and validate metadata
    saved_meta = load_metadata(persist_path)
    if not saved_meta:
        return False
    
    current_meta = get_index_metadata(pdf_path, chunk_size, chunk_overlap, embed_model)
    
    # Validate all parameters match
    return (
        saved_meta.get("pdf_fingerprint") == current_meta["pdf_fingerprint"] and
        saved_meta.get("chunk_size") == current_meta["chunk_size"] and
        saved_meta.get("chunk_overlap") == current_meta["chunk_overlap"] and
        saved_meta.get("embedding_model") == current_meta["embedding_model"]
    )

def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into context string."""
    return "\n\n".join(d.page_content for d in docs)

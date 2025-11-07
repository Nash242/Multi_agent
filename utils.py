from pathlib import Path
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas import RAGState
from helper import (
    load_pdf,
    split_documents,
    build_qdrant_vectorstore,
    load_qdrant_vectorstore,
    save_metadata,
    generate_collection_name,
    get_index_metadata,
    is_index_valid,
    format_documents
)
from config import INDEX_ROOT, LLM_MODEL, LLM_TEMPERATURE

@traceable(name="check_cache_node", tags=["cache"])
def check_cache_node(state: RAGState) -> dict:
    """Check if valid cached Qdrant collection exists."""
    # Generate collection name based on PDF hash only
    collection_name = generate_collection_name(state["pdf_path"])
    persist_dir = INDEX_ROOT / collection_name
    
    # Check if index is valid for current parameters
    cache_hit = (
        is_index_valid(
            collection_name,
            str(persist_dir),
            state["pdf_path"],
            state["chunk_size"],
            state["chunk_overlap"],
            state["embed_model"]
        )
        and not state.get("force_rebuild", False)
    )
    
    status = "✓ Found valid cached index" if cache_hit else "✗ No valid cache found"
    
    return {
        "collection_name": collection_name,
        "persist_dir": str(persist_dir),
        "cache_hit": cache_hit,
        "steps": [status]
    }

@traceable(name="load_pdf_node", tags=["loader"])
def load_pdf_node(state: RAGState) -> dict:
    """Load PDF document."""
    docs = load_pdf(state["pdf_path"])
    return {
        "docs": docs,
        "steps": [f"✓ Loaded PDF ({len(docs)} pages)"]
    }

@traceable(name="split_docs_node", tags=["splitter"])
def split_docs_node(state: RAGState) -> dict:
    """Split documents into chunks."""
    splits = split_documents(
        state["docs"],
        chunk_size=state["chunk_size"],
        chunk_overlap=state["chunk_overlap"]
    )
    return {
        "splits": splits,
        "steps": [f"✓ Split into {len(splits)} chunks"]
    }

@traceable(name="build_index_node", tags=["index", "build"])
def build_index_node(state: RAGState) -> dict:
    """Build new Qdrant vectorstore index."""
    vectorstore = build_qdrant_vectorstore(
        state["splits"],
        state["collection_name"],
        state["persist_dir"],
        state["embed_model"]
    )
    
    # Save metadata with all parameters
    metadata = get_index_metadata(
        state["pdf_path"],
        state["chunk_size"],
        state["chunk_overlap"],
        state["embed_model"]
    )
    metadata["collection_name"] = state["collection_name"]
    save_metadata(Path(state["persist_dir"]), metadata)
    
    return {
        "vectorstore": vectorstore,
        "steps": ["✓ Built and saved new Qdrant index"]
    }

@traceable(name="load_index_node", tags=["index", "load"])
def load_index_node(state: RAGState) -> dict:
    """Load existing Qdrant vectorstore from cache."""
    vectorstore = load_qdrant_vectorstore(
        state["collection_name"],
        state["persist_dir"],
        state["embed_model"]
    )
    
    return {
        "vectorstore": vectorstore,
        "steps": ["⚡ Loaded cached index (fast path)"]
    }

@traceable(name="retrieve_node", tags=["retrieval"])
def retrieve_node(state: RAGState) -> dict:
    """Retrieve relevant documents for the question."""
    retriever = state["vectorstore"].as_retriever(
        search_type="similarity",
        search_kwargs={"k": state.get("k", 4)}
    )
    
    docs = retriever.invoke(state["question"])
    context = format_documents(docs)
    
    return {
        "retrieved_docs": docs,
        "context": context,
        "steps": [f"✓ Retrieved {len(docs)} relevant chunks"]
    }

@traceable(name="generate_answer_node", tags=["generation"])
def generate_answer_node(state: RAGState) -> dict:
    """Generate answer using LLM with retrieved context."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Answer ONLY from the provided context. "
         "If the answer cannot be found in the context, clearly state that you don't know."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "question": state["question"],
        "context": state["context"]
    })
    
    return {
        "answer": answer,
        "steps": ["✓ Generated answer"]
    }

def should_build_index(state: RAGState) -> str:
    """Conditional edge: decide whether to build or load index."""
    return "load_index" if state["cache_hit"] else "load_pdf"


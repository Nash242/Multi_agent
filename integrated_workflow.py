from langgraph.graph import StateGraph, START, END

from unified_schemas import UnifiedState
from routing_nodes import router_node
from weather_nodes import weather_node, unknown_node
from utils import (
    check_cache_node,
    load_pdf_node,
    split_docs_node,
    build_index_node,
    load_index_node,
    retrieve_node,
    generate_answer_node,
    should_build_index
)

def route_after_classification(state: dict) -> str:
    """Conditional routing after agent classification."""
    agent_type = state.get("agent_type", "unknown")
    
    if agent_type == "weather":
        return "weather"
    elif agent_type == "rag":
        if not state.get("pdf_path"):
            return "unknown"  # No PDF available, can't do RAG
        return "check_cache"
    else:
        return "unknown"

def build_integrated_workflow():
    """Build integrated workflow with RAG and Weather agents."""
    graph = StateGraph(UnifiedState)
    
    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("weather", weather_node)
    graph.add_node("unknown", unknown_node)
    
    # RAG nodes (from your existing workflow)
    graph.add_node("check_cache", check_cache_node)
    graph.add_node("load_pdf", load_pdf_node)
    graph.add_node("split_docs", split_docs_node)
    graph.add_node("build_index", build_index_node)
    graph.add_node("load_index", load_index_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_answer_node)
    
    # Entry point: Route to agent
    graph.add_edge(START, "router")
    
    # Conditional routing after classification
    graph.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "weather": "weather",
            "check_cache": "check_cache",
            "unknown": "unknown"
        }
    )
    
    # Weather and unknown agents go directly to end
    graph.add_edge("weather", END)
    graph.add_edge("unknown", END)
    
    # RAG workflow (unchanged from your original)
    graph.add_conditional_edges(
        "check_cache",
        should_build_index,
        {
            "load_index": "load_index",
            "load_pdf": "load_pdf"
        }
    )
    
    graph.add_edge("load_pdf", "split_docs")
    graph.add_edge("split_docs", "build_index")
    graph.add_edge("build_index", "retrieve")
    graph.add_edge("load_index", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()


from typing import Dict
from langsmith import traceable
from agent_router import route_to_agent

@traceable(name="router_node", tags=["routing"])
def router_node(state: Dict) -> Dict:
    """Route to appropriate agent."""
    pdf_available = bool(state.get("pdf_path"))
    agent_type = route_to_agent(state["question"], pdf_available)
    
    return {
        "agent_type": agent_type,
        "steps": [f"🧭 Routed to: {agent_type.upper()} agent"]
    }


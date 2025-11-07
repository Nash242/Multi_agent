Multi-Agent Chatbot — RAG + Weather Assistant

An intelligent multi-agent chatbot that combines Retrieval-Augmented Generation (RAG) and real-time weather insights in a single unified system.
Ask questions about your uploaded PDFs or get live weather updates for any city — powered by LangChain, LangGraph, Qdrant, and OpenAI.

🌟 Features

✅ Document Q&A (RAG) — Upload any PDF, and the chatbot builds a Qdrant vector index for intelligent retrieval and context-aware answers.
✅ Weather Assistant — Ask about the current temperature, humidity, or weather conditions for any city worldwide.
✅ Multi-Agent Routing — Smartly routes your query to the correct agent (RAG or Weather) using LLM-based classification.
✅ Persistent Caching — Keeps your vector store and PDF summaries available across sessions.
✅ LangSmith Tracing — End-to-end observability and debugging for every LLM call.
✅ Streamlit UI + CLI — Use the chatbot via an interactive web app or terminal.

🧠 Architecture Overview
User Query
    │
    ▼
┌─────────────────────────────┐
│ 🔀 Agent Router (LLM)       │ → Classifies query as [RAG | Weather | Unknown]
└─────────────────────────────┘
         │
 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼                ▼
📄 RAG Agent   🌤️ Weather Agent   ❓ Fallback
 - PDF Loader   - City/State      - Handles unknown
 - Chunking     - OpenWeather API   questions gracefully
 - Qdrant Index
 - LLM Answering

🧩 Tech Stack
Component	Technology
LLM	OpenAI GPT (via langchain-openai)
Framework	LangChain + LangGraph
Vector Store	Qdrant (local persistent mode)
Frontend	Streamlit
Tracing & Debugging	LangSmith
Environment Management	Python + dotenv
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/multi-agent-chatbot.git
cd multi-agent-chatbot

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3️⃣ Install Dependencies
pip install -U langchain langchain-openai langchain-community langchain-qdrant \
qdrant-client pypdf python-dotenv langgraph streamlit requests langsmith

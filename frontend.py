import streamlit as st
import atexit
import hashlib
from pathlib import Path
from integrated_app import IntegratedApp
from helper import cleanup_clients


st.set_page_config(page_title="Multi-Agent Assistant", page_icon="📄", layout="wide")

# Register cleanup
atexit.register(cleanup_clients)

# Initialize app (cached)
@st.cache_resource
def get_app():
    return IntegratedApp()

app = get_app()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

def get_file_hash(file_bytes):
    """Get hash of file content."""
    return hashlib.sha256(file_bytes).hexdigest()


# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    
    if uploaded_file:
        # Read file bytes once
        file_bytes = uploaded_file.read()
        current_hash = get_file_hash(file_bytes)
        
        # Check if this is a new file
        if st.session_state.pdf_hash != current_hash:
            # Save uploaded file with unique name based on hash
            pdf_path = Path(f"uploaded_pdfs/pdf_{current_hash[:16]}.pdf")
            pdf_path.parent.mkdir(exist_ok=True)
            pdf_path.write_bytes(file_bytes)
            
            # Update session state
            st.session_state.pdf_hash = current_hash
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_path = str(pdf_path)
            st.session_state.pdf_uploaded = True
            st.session_state.messages = []  # Clear chat history for new PDF
            st.success(f"✓ Loaded: {uploaded_file.name}")
        else:
            st.success(f"✓ Using: {st.session_state.pdf_name}")

    
    # Show cache status
    if st.session_state.pdf_uploaded:
        st.subheader("📊 Status")
        st.metric("Current PDF", st.session_state.pdf_name)
        st.metric("Questions Asked", len([m for m in st.session_state.messages if m['role'] == 'user']))
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("♻️ Cleanup Clients"):
        app.cleanup()
        st.success("Qdrant clients cleaned up!")
    
    st.divider()


# Main chat interface
st.title("📄 PDF Question Answering with RAG")

# Display PDF status
if st.session_state.pdf_uploaded and st.session_state.pdf_name:
    st.info(f"📄 Currently chatting with: **{st.session_state.pdf_name}**")
else:
    st.warning("👈 Please upload a PDF file to start chatting")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    st.text(source["content"])
                    with st.expander("Metadata"):
                        st.json(source["metadata"])
                    if i < len(message["sources"]):
                        st.divider()

# Chat input (enabled always)
prompt = st.chat_input("Ask about weather or your documents...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                import time
                start_time = time.time()

                # Detect if PDF is available
                pdf_path = st.session_state.pdf_path if st.session_state.pdf_uploaded else None

                # Call your integrated app
                result = app.query(
                    question=prompt,
                    pdf_path=pdf_path,   # <-- None for weather/general
                    chunk_size=1000,
                    chunk_overlap=150,
                    k=3
                )

                elapsed = time.time() - start_time

                # Display answer
                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)

                # Metadata
                agent_type = result.get("agent_type", result.get("metadata", {}).get("agent", "unknown"))
                st.caption(f"🤖 Agent: **{agent_type.upper()}** • ⏱ {elapsed:.2f}s")

                # If weather
                if agent_type == "weather" and result.get("metadata", {}).get("success"):
                    weather = result["weather_data"]
                    with st.expander("🌤️ Weather Details"):
                        st.json({
                            "city": result.get("city"),
                            "state": result.get("state"),
                            "temperature": weather["main"]["temp"],
                            "feels_like": weather["main"]["feels_like"],
                            "humidity": weather["main"]["humidity"],
                            "description": weather["weather"][0]["description"],
                            "wind_speed": weather["wind"]["speed"]
                        })

                # If RAG
                elif agent_type == "rag":
                    sources = app.get_sources(result)
                    cache_status = "⚡ Cached" if result.get("cache_hit") else "🔨 New Index"
                    st.caption(f"{cache_status} • {len(result.get('retrieved_docs', []))} docs")

                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}**")
                            st.text(source["content"])
                            with st.expander("Metadata"):
                                st.json(source["metadata"])
                            if i < len(sources):
                                st.divider()

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": result.get("metadata", {}),
                    "agent_type": agent_type,
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# Chat input
# if st.session_state.pdf_uploaded and st.session_state.pdf_path:
#     if prompt := st.chat_input("Ask a question about the document..."):
#         # Add user message to chat
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     import time
#                     start_time = time.time()
                    
#                     result = app.query(
#                         pdf_path=st.session_state.pdf_path,
#                         question=prompt,
#                         chunk_size=1000,
#                         chunk_overlap=150,
#                         k=3
#                     )
                    
#                     elapsed = time.time() - start_time
                    
#                     # Display answer
#                     answer = result["answer"]
#                     st.markdown(answer)
                    
#                     # Show processing info
#                     cache_status = "⚡ Cached" if result.get("cache_hit") else "🔨 Built new"
#                     st.caption(f"{cache_status} • {len(result.get('retrieved_docs', []))} docs • {elapsed:.1f}s")
                    
#                     # Get sources
#                     sources = app.get_sources(result)
                    
#                     # Show sources in expander
#                     with st.expander("📚 View Sources"):
#                         for i, source in enumerate(sources, 1):
#                             st.markdown(f"**Source {i}**")
#                             st.text(source["content"])
#                             with st.expander("Metadata"):
#                                 st.json(source["metadata"])
#                             if i < len(sources):
#                                 st.divider()
                    
#                     # Add assistant message to chat with sources
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "content": answer,
#                         "sources": sources
#                     })
                    
#                 except Exception as e:
#                     error_msg = f"❌ Error: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "content": error_msg
#                     })
# else:
#     st.chat_input("Upload a PDF to start chatting from the pdf...")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("💬 Chat History")
    st.caption(f"{len([m for m in st.session_state.messages if m['role'] == 'user'])} questions asked")
with col2:
    st.caption("🤖 Powered by")
    st.caption("Qdrant + LangChain")
with col3:
    st.caption("⚡ Performance")
    st.caption("Cached queries < 2s")


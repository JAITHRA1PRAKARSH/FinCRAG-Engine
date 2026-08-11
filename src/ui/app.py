import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="FinCRAG - SEC Financial Analyst",
    page_icon="📈",
    layout="wide"
)

# Cache the Agent loading so model downloads only happen ONCE per server boot
@st.cache_resource(show_spinner="Booting hybrid retriever & vector database...")
def load_agent():
    from src.agent.crag_agent import app as agent_app
    return agent_app

agent_app = load_agent()

st.title("📈 FinCRAG: Corrective RAG Financial Agent")
st.caption("Powered by LangGraph, Qdrant Hybrid Search, and Groq Llama 3.3 70B")

# Sidebar Details
with st.sidebar:
    st.header("About FinCRAG")
    st.markdown("""
    This engine extracts, chunk-indexes, and analyzes SEC 10-K filings for **Apple Inc. (AAPL)** and **Microsoft Corp. (MSFT)**.
    
    **Features:**
    - Table-aware Markdown chunking
    - Hybrid Search (BM25 + Dense Vectors)
    - Cross-Encoder Re-Ranking
    - Self-Correcting Reflection Graph
    """)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Memory in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your SEC Financial Analyst. Ask me anything about Apple or Microsoft's latest 10-K filings."}
    ]

# Display Previous Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle User Input
if user_query := st.chat_input("e.g., Compare supply chain risks between Apple and Microsoft"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC filings and grading relevance..."):
            try:
                result = agent_app.invoke({
                    "original_question": user_query, 
                    "question": user_query, 
                    "loop_count": 0
                })
                
                answer = result.get("answer", "No answer returned.")
                loops = result.get("loop_count", 0)
                
                full_response = f"{answer}\n\n---\n*`[Agent Executed in {loops} self-correction loop(s)]`*"
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Failed to process query: {e}")
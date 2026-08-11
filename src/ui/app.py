import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

# Streamlit Page Setup
st.set_page_config(
    page_title="FinCRAG - SEC Financial Analyst",
    page_icon="📈",
    layout="wide"
)

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
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Show loading spinner while calling FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC filings and grading relevance..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": user_query},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    loops = data.get("loops_taken", 0)
                    
                    # Add execution metadata badge
                    full_response = f"{answer}\n\n---\n*`[Agent Executed in {loops} self-correction loop(s)]`*"
                    st.write(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    err_msg = f"API Error: Server returned status code {response.status_code}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {e}")
                
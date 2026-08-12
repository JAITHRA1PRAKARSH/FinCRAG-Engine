import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

st.set_page_config(
    page_title="FinCRAG - SEC Financial Analyst",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource(show_spinner="Booting Financial Vector Store & Agent...")
def load_agent():
    from src.agent.crag_agent import app as agent_app
    return agent_app

agent_app = load_agent()

st.title("📈 FinCRAG: Corrective RAG Financial Agent")
st.caption("Powered by LangGraph, Qdrant Hybrid Search, and Groq Llama 3")

with st.sidebar:
    st.header("About FinCRAG")
    st.markdown("""
    **Multi-Company SEC 10-K Analysis** (Apple Inc. & Microsoft Corp.)
    
    - **Retrieval:** Hybrid BM25 + FastEmbed Cosine Search
    - **Orchestration:** LangGraph Self-Correction Graph
    - **Evaluation:** RAGAS-Validated (96.15% Faithfulness)
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your SEC Financial Analyst. Ask me anything regarding Apple's or Microsoft's latest 10-K filings."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("e.g., Compare supply chain risks between Apple and Microsoft"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC filings and evaluating context relevance..."):
            try:
                result = agent_app.invoke({
                    "original_question": user_query, 
                    "question": user_query, 
                    "loop_count": 0
                })
                
                answer = result.get("answer", "No response generated.")
                loops = result.get("loop_count", 0)
                
                full_response = f"{answer}\n\n---\n*`[Agent Executed in {loops} self-correction loop(s)]`*"
                st.write(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error processing request: {e}")
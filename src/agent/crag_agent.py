import os
import gc
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.ingestion.chunker import load_all_filings
from src.retrieval.hybrid_search import HybridRetriever

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from environment/secrets!")

# 1. State Definition
class AgentState(TypedDict):
    original_question: str
    question: str
    documents: List[str]
    is_relevant: str
    answer: str
    loop_count: int

# 2. Tiered LLMs Setup
# Fast 8B model for background grading & rewriting (saves ~70% of 70B token quota)
fast_llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
# High-reasoning 70B model for final cited answer generation
synthesis_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

class GraderOutput(BaseModel):
    binary_score: str = Field(description="Score 'yes' if context is relevant, 'no' otherwise.")

structured_grader = fast_llm.with_structured_output(GraderOutput)

# 3. Retriever Initialization
_retriever = None

def get_agent_retriever():
    global _retriever
    if _retriever is None:
        chunks = load_all_filings()
        _retriever = HybridRetriever(chunks)
        del chunks
        gc.collect()
    return _retriever

# 4. Agent Nodes
def retrieve_node(state: AgentState):
    loop_count = state.get("loop_count", 0)
    current_q = state.get("question", state["original_question"])
    search_k = 6 if loop_count == 0 else 10
    
    retriever = get_agent_retriever()
    docs = retriever.search(current_q, final_k=search_k)
    doc_texts = [d.page_content for d in docs]
    return {"documents": doc_texts, "question": current_q, "loop_count": loop_count}

def grade_documents_node(state: AgentState):
    question = state["original_question"]
    documents = state["documents"]
    context = "\n\n".join(documents)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict financial auditor grading retrieved SEC context.
        Guidelines:
        - For NUMERIC queries: score 'yes' if relevant tables/numbers are present.
        - For QUALITATIVE queries: score 'yes' if descriptive risk factors or disclosures are present.
        Return 'yes' if the context can directly answer the question; otherwise return 'no'."""),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    try:
        result = (prompt | structured_grader).invoke({"question": question, "context": context})
        score = result.binary_score.lower()
    except Exception:
        score = "yes"
        
    return {"is_relevant": score}

def decide_to_generate(state: AgentState):
    if state.get("is_relevant") == "yes" or state.get("loop_count", 0) >= 2:
        return "generate"
    return "rewrite"

def rewrite_query_node(state: AgentState):
    original_q = state["original_question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an SEC search query rewriter. 
        Analyze the question intent and generate an optimized query targeting SEC 10-K terms (e.g. 'Consolidated Financial Statements', 'Item 1A Risk Factors').
        Return ONLY the rewritten query text."""),
        ("human", "{question}")
    ])
    
    try:
        new_q = (prompt | fast_llm).invoke({"question": original_q}).content.replace('"', '').strip()
    except Exception:
        new_q = original_q
        
    loop_count = state.get("loop_count", 0) + 1
    return {"question": new_q, "loop_count": loop_count}

def generate_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an SEC Financial Analyst. Answer the user question using ONLY the provided context.
        
        CITATION INSTRUCTIONS:
        - For every key claim, figure, or risk factor, supply an explicit inline citation with the filing name and section (e.g. [Apple 10-K: Item 1A] or [Microsoft 10-K: Consolidated Statements of Operations]).
        - If the provided context does not contain the answer, explicitly state: 'I cannot find this information in the filings.'
        - Never hallucinate data.
        
        CONTEXT:
        {context}"""),
        ("human", "{question}")
    ])
    
    chain = prompt | synthesis_llm
    response = chain.invoke({
        "context": "\n\n---\n\n".join(state["documents"]),
        "question": state["original_question"]
    })
    return {"answer": response.content}

# 5. Compile Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_to_generate, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()
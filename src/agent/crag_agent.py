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

# 2. LLM Setup
print("[*] Initializing Llama 3.3 on Groq...")
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

class GraderOutput(BaseModel):
    binary_score: str = Field(description="Score 'yes' if context is relevant, 'no' otherwise.")

structured_grader = llm.with_structured_output(GraderOutput)

# 3. Hybrid Retriever Singleton Setup
print("[*] Loading filings and building retriever...")
chunks = load_all_filings()
retriever = HybridRetriever(chunks)
del chunks  # Free raw chunk list from RAM immediately
gc.collect()

# 4. Agent Nodes
def retrieve_node(state: AgentState):
    loop_count = state.get("loop_count", 0)
    current_q = state.get("question", state["original_question"])
    search_k = 6 if loop_count == 0 else 10
    
    print(f"\n[Agent] Action: Retrieving (k={search_k}) for query: '{current_q}'")
    docs = retriever.search(current_q, final_k=search_k)
    doc_texts = [d.page_content for d in docs]
    return {"documents": doc_texts, "question": current_q, "loop_count": loop_count}

def grade_documents_node(state: AgentState):
    print("[Agent] Action: Grading document relevance...")
    question = state["original_question"]
    documents = state["documents"]
    
    context = "\n\n".join(documents)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict SEC auditor grading retrieved context relevance for a user question.
        
        Evaluation Guidelines:
        1. If the question asks for NUMERIC data (e.g., net sales, revenues, expenses), score 'yes' ONLY if exact numbers/tables are present in the context.
        2. If the question asks for QUALITATIVE info (e.g., supply chain risks, tariffs, business drivers, policies), score 'yes' if relevant descriptive text or risk factor explanations are present.
        
        Return 'yes' if the context contains enough relevant information to directly answer or address the question. Otherwise, return 'no'."""),
        ("human", "Question: {question}\n\nRetrieved Context:\n{context}")
    ])
    
    try:
        result = (prompt | structured_grader).invoke({"question": question, "context": context})
        score = result.binary_score.lower()
    except Exception:
        score = "yes"
        
    print(f"[Agent] Grade Result: '{score}'")
    return {"is_relevant": score}

def decide_to_generate(state: AgentState):
    if state.get("is_relevant") == "yes":
        return "generate"
    elif state.get("loop_count", 0) >= 2:
        return "generate"
    else:
        return "rewrite"

def rewrite_query_node(state: AgentState):
    print("[Agent] Action: Rewriting query for improved retrieval...")
    original_q = state["original_question"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert SEC search query generator. 
        Analyze the intent of the question and rewrite it to maximize retrieval performance from an SEC 10-K document:
        
        - If the question asks for FINANCIAL NUMBERS (revenue, sales, expenses): target exact financial table terms (e.g., 'Consolidated Statements of Operations table', 'numeric net sales').
        - If the question asks for RISKS or STRATEGY (supply chain, tariffs, competition): target SEC section terms (e.g., 'Item 1A Risk Factors', 'supply chain disruptions', 'trade restrictions and tariffs').
        
        Output ONLY the rewritten search query with no quotes or additional explanation."""),
        ("human", "{question}")
    ])
    
    new_q = (prompt | llm).invoke({"question": original_q}).content.replace('"', '').strip()
    loop_count = state.get("loop_count", 0) + 1
    return {"question": new_q, "loop_count": loop_count}

def generate_node(state: AgentState):
    print("[Agent] Action: Generating final answer with citations...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly accurate SEC Financial Analyst. 
        Answer the user's question using ONLY the provided context. 
        
        CITATION INSTRUCTIONS:
        - For every key claim, risk factor, or numerical figure, provide an explicit inline citation indicating which company filing and section it came from (e.g., [Apple 10-K: Item 1A - Risk Factors] or [Microsoft 10-K: Consolidated Statements of Operations]).
        - If the context does not contain the answer, state: 'I cannot find this information in the filings.'
        - Do not invent citations or hallucinate numbers.
        
        CONTEXT:
        {context}"""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
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
import os
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Import custom ingestion and retrieval pipeline
from src.ingestion.chunker import load_all_filings
from src.retrieval.hybrid_search import HybridRetriever

# Load API key from .env file
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env file!")

# 1. Define the State (The agent's memory)
class AgentState(TypedDict):
    original_question: str
    question: str
    documents: List[str]
    is_relevant: str
    answer: str
    loop_count: int

# 2. Setup LLM and Retriever
print("[*] Initializing Llama 3.3 on Groq...")
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Setup the JSON Grader for the Evaluation Node
class GraderOutput(BaseModel):
    binary_score: str = Field(description="Score 'yes' if the context contains enough relevant information to directly answer or address the question. Score 'no' otherwise.")

structured_grader = llm.with_structured_output(GraderOutput)

print("[*] Booting up Hybrid Retrieval Engine across all filings...")
chunks = load_all_filings()
retriever = HybridRetriever(chunks)


# 3. Define the Agent Nodes

def retrieve_node(state: AgentState):
    """Node 1: Retrieves chunks based on the current question."""
    loop_count = state.get("loop_count", 0)
    current_q = state.get("question", state["original_question"])
    
    # Retrieve 6 chunks to ensure multi-company context is retrieved
    search_k = 6 if loop_count == 0 else 10
    
    print(f"\n[Agent] Action: Retrieving (k={search_k}) for query: '{current_q}'")
    docs = retriever.search(current_q, final_k=search_k)
    doc_texts = [d.page_content for d in docs]
    
    return {"documents": doc_texts, "question": current_q, "loop_count": loop_count}

def grade_documents_node(state: AgentState):
    """Node 2: Evaluates if the context effectively addresses the query."""
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
    except Exception as e:
        score = "yes"  # Failsafe
        
    print(f"[Agent] Grade Result: '{score}'")
    return {"is_relevant": score}

def decide_to_generate(state: AgentState):
    """Conditional Edge: Decides whether to generate or rewrite."""
    if state.get("is_relevant") == "yes":
        print("[Agent] Decision: Context is good. Proceeding to Generation.")
        return "generate"
    elif state.get("loop_count", 0) >= 2:
        print("[Agent] Decision: Max loops reached. Forcing Generation.")
        return "generate"
    else:
        print("[Agent] Decision: Context is missing required details. Triggering Self-Correction!")
        return "rewrite"

def rewrite_query_node(state: AgentState):
    """Node 3: Dynamically rewrites queries based on intent."""
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
    
    print(f"[Agent] Rewritten Query: '{new_q}'")
    return {"question": new_q, "loop_count": loop_count}

def generate_node(state: AgentState):
    """Node 4: Generates the final answer with strict inline citations."""
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

# 4. Build the LangGraph
print("[*] Compiling CRAG Workflow...")
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINCRAG AGENT INITIALIZED - TESTING SELF-CORRECTION")
    print("="*70)
    
    test_question = "Compare the primary supply chain risks and international trade concerns mentioned by Apple versus Microsoft."
    
    result = app.invoke({
        "original_question": test_question, 
        "question": test_question, 
        "loop_count": 0
    })
    
    print("\n" + "="*70)
    print("FINAL AGENT ANSWER:")
    print("="*70)
    print(result["answer"])
    print("="*70)
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Import our exact LangGraph Agent
from src.agent.crag_agent import app as agent_app

def run_evaluation():
    print("\n[*] Generating Agent Responses for Evaluation Dataset...")
    
    # 1. Define test questions and the IDEAL "Ground Truth" answers
    questions = [
        "What was Apple's total net sales by reportable segment for 2025?",
        "Compare the primary supply chain risks mentioned by Apple versus Microsoft."
    ]
    
    ground_truths = [
        "Apple's net sales by segment in 2025 were: Americas $178,353 million, Europe $111,032 million, Greater China $64,377 million, Japan $28,703 million, and Rest of Asia Pacific $29,663 million.",
        "Microsoft explicitly identifies international trade risks, including U.S. tariffs, shifting AI export controls, and sanctions. Apple focuses more broadly on macroeconomic risks, industrial accidents, and public health issues rather than specific international trade tariffs."
    ]
    
    answers = []
    contexts = []
    
    # 2. Run agent for each question to get its live answers and retrieved chunks
    for q in questions:
        print(f"    -> Interrogating Agent: '{q}'")
        result = agent_app.invoke({
            "original_question": q, 
            "question": q, 
            "loop_count": 0
        })
        answers.append(result["answer"])
        contexts.append(result["documents"])
        
    # 3. Format as a Dataset
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Initialize Evaluation Judges
    print("\n[*] Initializing RAGAS AI Judges...")
    eval_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    eval_embeddings = FastEmbedEmbeddings()
    
    # 5. Run RAGAS
    print("[*] Running RAGAS Scientific Evaluation... (This takes 30-60 seconds)")
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    # 6. Print the metrics report
    print("\n" + "="*80)
    print("🎯 RAGAS EVALUATION METRICS REPORT:")
    print("="*80)
    
    df = result.to_pandas()
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df)
    
    print("\n--- Final Aggregated Engine Scores ---")
    print(result)
    print("================================================================================")

if __name__ == "__main__":
    run_evaluation()
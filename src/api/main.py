from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Import our compiled CRAG LangGraph agent
from src.agent.crag_agent import app as agent_app

app = FastAPI(title="FinCRAG SEC API", version="1.0")

# Define the data structure for incoming requests
class ChatRequest(BaseModel):
    query: str

# Define the data structure for outgoing responses
class ChatResponse(BaseModel):
    answer: str
    loops_taken: int

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    print(f"\n[API] Received Query: {request.query}")
    
    # Invoke the LangGraph agent
    result = agent_app.invoke({
        "original_question": request.query,
        "question": request.query,
        "loop_count": 0
    })
    
    return ChatResponse(
        answer=result["answer"],
        loops_taken=result["loop_count"]
    )

if __name__ == "__main__":
    print("[*] Starting FinCRAG API Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List, Optional

from app.chat import stream_chat
from app.rag import ingest_document, query_knowledge_base, run_automated_eval, pre_populate_docs, get_ingested_documents
from app.agent import run_planning_agent
from app.coder import generate_and_heal_code

app = FastAPI(title="AI Platform")

@app.on_event("startup")
async def startup_event():
    """Pre-populates the document database on application startup."""
    print("Pre-populating documents...")
    pre_populate_docs()
    print("Documents pre-populated.")

# --- Models ---
class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    url: str
    name: str

class QueryRequest(BaseModel):
    query: str

class AgentRequest(BaseModel):
    prompt: str

# --- Routes ---

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Task 3.1: Streaming Chat"""
    return StreamingResponse(stream_chat(req.message), media_type="application/x-ndjson")

@app.post("/rag/ingest")
async def rag_ingest(req: IngestRequest):
    """Task 3.2a: Ingest"""
    result = ingest_document(req.url, req.name)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/rag/query")
async def rag_query(req: QueryRequest):
    """Task 3.2c: Query"""
    return query_knowledge_base(req.query)

@app.post("/rag/eval")
async def rag_eval():
    """Task 3.2d: Eval"""
    return run_automated_eval()

@app.get("/rag/ingested-docs")
async def rag_ingested_docs():
    """Returns the list of ingested documents."""
    return {"documents": get_ingested_documents()}

@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    """Task 3.3: Agent"""
    plan, logs = run_planning_agent(req.prompt)
    return {"plan": plan, "logs": logs}

@app.post("/coder")
async def coder_endpoint(req: AgentRequest):
    """Task 3.4: Coder Stream"""
    def generator():
        for update in generate_and_heal_code(req.prompt):
            yield update + "\n"
    return StreamingResponse(generator(), media_type="text/plain")

import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent_brain import compiled_local_agent
from ingestion import run_local_ingestion

app = FastAPI(title="Edge Agentic RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = "./_server_cache"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


# --- DATA SCHEMAS ---
class QueryInput(BaseModel):
    prompt: str = Field(..., min_length=1)

class AgentResponse(BaseModel):
    status: str
    response: str


# --- ENDPOINT 1: THE INGESTION GATEWAY ---
@app.post("/upload-resume")
async def handle_resume_upload(file: UploadFile = File(...)):
    """Catches a raw binary PDF from the web, saves it temporarily, and triggers GPU indexing."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Strictly .pdf or .txt accepted.")

    temp_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

    # Stream incoming binary stream directly to hard drive
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_created = run_local_ingestion(temp_path)
        return {
            "status": "success",
            "file_processed": file.filename,
            "vector_count": chunks_created,
            "system_message": "Candidate successfully vectorized into local ChromaDB."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Crash: {str(e)}")
    finally:
        # Garbage collection: delete the raw file so the server hard drive doesn't fill up
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- ENDPOINT 2: THE AGENT BRAIN ---
@app.post("/analyze", response_model=AgentResponse)
async def run_agent_analysis(payload: QueryInput):
    try:
        print(f"\n📥 INCOMING REQUEST: '{payload.prompt}'")
        final_output = compiled_local_agent.invoke({"messages": [("user", payload.prompt)]})
        
        # If Qwen extracted clean JSON, grab it; otherwise grab standard chat text
        if final_output.get("extracted_json"):
            reply = f"```json\n{final_output['extracted_json']}\n```"
        else:
            reply = final_output["messages"][-1].content
            
        return AgentResponse(status="success", response=reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

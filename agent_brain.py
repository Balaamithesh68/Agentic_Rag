import json
import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

# Grab the container network URL
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 1. DEFINE THE STRICT JSON SCHEMA (Pydantic)
class ExtractedResumeData(BaseModel):
    candidate_name: str = Field(description="The full legal name of the candidate")
    primary_role: str = Field(description="The job title or role they are applying for")
    technical_skills: list[str] = Field(description="List of all programming languages, frameworks, and tools")
    years_of_experience: int = Field(
        default=0, 
        description="Total estimated years of professional work experience. Must be a realistic single digit integer between 0 and 15."
    )

    @field_validator('years_of_experience', mode='before')
    @classmethod
    def clean_experience_number(cls, raw_value):
        try:
            val = int(raw_value)
            if val > 50 or val < 0:
                print(f"⚠️ Intercepted hallucinated integer ({val}). Defaulting to 0.")
                return 0
            return val
        except (ValueError, TypeError):
            return 0

# 2. DEFINE THE GRAPH STATE
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[str]
    routing_decision: str
    extracted_json: dict
    error_feedback :str
    retry_count: int

# 3. INITIALIZE ENGINES WITH base_url
print(f"🔗 Initializing Brain Base Models targeting: {OLLAMA_URL}")
llm = ChatOllama(
    model="qwen2.5:3b", 
    temperature=0, 
    base_url=OLLAMA_URL,
    keep_alive="24h"  # <-- "-1" forces Ollama to keep weights in GPU VRAM forever
)
embeddings_engine = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_engine,
    collection_name="resume_skills_collection"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. DEFINE THE ARCHITECTURAL NODES
def intent_router_node(state: GraphState):
    print("🤖 NODE: Analyzing User Intent...")
    last_message = state["messages"][-1].content
    
    system_prompt = (
        "You are an API router. Analyze the user query. "
        "If the query asks to extract, parse, list, or format the candidate's skills or resume into JSON or a structured format, reply with exactly 'EXTRACT'. "
        "If the query asks general background questions requiring lookup, reply with exactly 'RETRIEVE'. "
        "If the query is conversational, reply with exactly 'DIRECT'."
    )
    
    response = llm.invoke([("system", system_prompt), ("user", last_message)])
    decision = response.content.strip().upper()
    print(f"🛣️ ROUTER DECISION: {decision}")
    return {"routing_decision": decision}

def retrieve_db_node(state: GraphState):
    print("🤖 NODE: Querying ChromaDB...")
    last_message = state["messages"][-1].content
    matched_docs = retriever.invoke(last_message)
    return {"documents": [doc.page_content for doc in matched_docs]}

def generate_answer_node(state: GraphState):
    print("🤖 NODE: Compiling Contextual Answer...")
    last_message = state["messages"][-1].content
    context = "\n---\n".join(state.get("documents", []))
    
    system_prompt = f"Answer using only the context:\n{context}"
    response = llm.invoke([("system", system_prompt), ("user", last_message)])
    return {"messages": [response]}

def skill_extraction_node(state: GraphState):
    current_retry = state.get("retry_count", 0)
    print(f"🤖 NODE: Executing Structured Extraction (Attempt {current_retry + 1}/3)...")
    
    all_docs = vector_store.get()
    full_resume_text = "\n".join(all_docs.get("documents", ["No resume found on disk."]))
    
    # 1. Check if we are in a self-correction loop
    feedback = state.get("error_feedback", "")
    if feedback:
        print("⚠️ FEEDBACK DETECTED: Forcing LLM to self-correct...")
        prompt = (
            f"You previously failed to extract the JSON. Here was the exact system error:\n"
            f"{feedback}\n\n"
            f"DO NOT make this mistake again. Extract the profile from this text:\n\n{full_resume_text}"
        )
    else:
        prompt = f"Extract the candidate profile from this resume text:\n\n{full_resume_text}"
    
    # 2. Execute and Catch
    try:
        structured_engine = llm.with_structured_output(ExtractedResumeData)
        parsed_data = structured_engine.invoke(prompt)
        
        data_dict = parsed_data.model_dump()
        formatted_string = f"```json\n{json.dumps(data_dict, indent=2)}\n```"
        
        # Success! Clear errors and pass the data forward.
        return {
            "extracted_json": data_dict,
            "messages": [("assistant", formatted_string)],
            "error_feedback": "", 
            "retry_count": current_retry + 1
        }
        
    except Exception as e:
        # Crash! Save the exact red error log to show to the LLM on the next loop.
        print(f"❌ EXTRACTION FAILED: {str(e)[:100]}...")
        return {
            "error_feedback": str(e),
            "retry_count": current_retry + 1
        }
def check_extraction_quality(state: GraphState):
    # If there are no errors, or we maxed out our 3 attempts, end the graph.
    if state.get("error_feedback") == "" or state.get("retry_count", 0) >= 3:
        if state.get("retry_count", 0) >= 3 and state.get("error_feedback") != "":
            print("🛑 FATAL: Agent failed 3 times. Terminating loop.")
        else:
            print("✅ QUALITY PASSED: JSON Schema is flawless.")
        return "end_path"
    
    # If there is an error, force the loop back to the extractor!
    print("🔄 QUALITY FAILED: Triggering Agentic Reflexion loop...")
    return "retry_path"

# 5. CONDITIONAL EDGE LOGIC
def route_conditional_path(state: GraphState):
    if state["routing_decision"] == "EXTRACT":
        return "extract_path"
    elif state["routing_decision"] == "RETRIEVE":
        return "retrieve_path"
    return "direct_path"

# 6. ASSEMBLE GRAPH
workflow = StateGraph(GraphState)

workflow.add_node("intent_router", intent_router_node)
workflow.add_node("retrieve_db", retrieve_db_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("skill_extractor", skill_extraction_node)

workflow.add_edge(START, "intent_router")

workflow.add_conditional_edges(
    "intent_router",
    route_conditional_path,
    {
        "extract_path": "skill_extractor",
        "retrieve_path": "retrieve_db",
        "direct_path": "generate_answer"
    }
)

workflow.add_edge("retrieve_db", "generate_answer")
workflow.add_edge("generate_answer", END)
# Replace workflow.add_edge("skill_extractor", END) with this:

workflow.add_conditional_edges(
    "skill_extractor",
    check_extraction_quality,
    {
        "end_path": END,
        "retry_path": "skill_extractor"  # <-- The cyclic loop!
    }
)

# THE CRITICAL EXPORT VARIABLE
compiled_local_agent = workflow.compile()

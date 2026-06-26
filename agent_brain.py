import json
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# 1. DEFINE THE STRICT JSON SCHEMA (Pydantic)
class ExtractedResumeData(BaseModel):
    candidate_name: str = Field(description="The full legal name of the candidate")
    primary_role: str = Field(description="The job title or role they are applying for")
    technical_skills: list[str] = Field(description="List of all programming languages, frameworks, and tools")
    years_of_experience: int = Field(default=0, description="Total estimated years of professional experience")

# 2. DEFINE THE GRAPH STATE
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[str]
    routing_decision: str
    extracted_json: dict  # <-- NEW: Holds our parsed JSON data

# 3. INITIALIZE ENGINES
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
embeddings_engine = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_engine,
    collection_name="resume_skills_collection"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. DEFINE THE NODES
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

# --- NEW NODE: THE PARSER ---
def skill_extraction_node(state: GraphState):
    """
    Forces Qwen 3B to output data strictly adhering to the Pydantic schema.
    """
    print("🤖 NODE: Executing Structured JSON Extraction...")
    
    # Grab the entire raw resume text straight out of ChromaDB storage
    all_docs = vector_store.get()
    full_resume_text = "\n".join(all_docs.get("documents", ["No resume found on disk."]))
    
    # .with_structured_output is a LangChain wrapper that binds the LLM to Pydantic
    structured_engine = llm.with_structured_output(ExtractedResumeData)
    
    prompt = f"Extract the candidate profile from this resume text:\n\n{full_resume_text}"
    parsed_data = structured_engine.invoke(prompt)
    
    # Convert Pydantic object back to a Python dictionary for state storage
    data_dict = parsed_data.model_dump()
    
    # Format a clean message string to send back to the user screen
    formatted_string = f"```json\n{json.dumps(data_dict, indent=2)}\n```"
    
    return {
        "extracted_json": data_dict,
        "messages": [("assistant", formatted_string)]
    }

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
workflow.add_node("skill_extractor", skill_extraction_node) # <-- NEW

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
workflow.add_edge("skill_extractor", END)

compiled_local_agent = workflow.compile()
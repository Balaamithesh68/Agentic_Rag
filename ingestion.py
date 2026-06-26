import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. SETUP ENGINE LOCAL VARIABLES
DB_DIR = "./chroma_db"
COLLECTION_NAME = "resume_skills_collection"
EMBEDDING_MODEL = "nomic-embed-text" # Ultra-lightweight local embedding model

def run_local_ingestion(file_path: str):
    """
    Reads a local text document, splits it into semantic overlapping chunks,
    generates math vectors using an offline model, and commits it to disk.
    """
    # Defensive check: ensure document source exists
    if not os.path.exists(file_path):
        print(f"❌ Ingestion aborted: The file '{file_path}' does not exist.")
        return

    print(f"📖 Reading source file: {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_document_text = f.read()

    # 2. CONFIGURATIVE SEMANTIC CHUNKING
    # We use RecursiveCharacterTextSplitter because it intelligently tries to split 
    # by paragraphs (\n\n), then sentences (\n), then words to maintain contextual coherence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Characters per text block
        chunk_overlap=100     # Preserves context hanging across boundaries
    )
    
    document_chunks = text_splitter.split_text(raw_document_text)
    print(f"✂️ Split document into {len(document_chunks)} unique structural chunks.")

    # 3. INITIALIZE LOCAL EMBEDDING PIPELINE
    # This reaches out to your running Ollama engine background service
    print(f"🧠 Generating vector embeddings via local model '{EMBEDDING_MODEL}'...")
    local_embeddings_engine = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. INSTANTIATE AND PERSIST DATABASE STACK
    print(f"💾 Saving vectors to persistent disk storage at: {DB_DIR}...")
    vector_store = Chroma.from_texts(
        texts=document_chunks,
        embedding=local_embeddings_engine,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("✅ Ingestion Pipeline complete! Database is locked, loaded, and fully offline.")
    return vector_store

if __name__ == "__main__":
    # Create a quick dummy file to test your pipeline execution
    test_file = "sample_resume.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Bala Amithesh S. is a Full-Stack Developer Intern candidate. "
                    "Expertise includes Python, FastAPI, LangGraph, and Machine Learning. "
                    "Built hybrid transformer mamba machine learning architectures for resume entity extraction.")
    
    run_local_ingestion(test_file)
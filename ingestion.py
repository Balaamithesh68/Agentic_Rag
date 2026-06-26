import os
import pypdf
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_DIR = "./chroma_db"
COLLECTION_NAME = "resume_skills_collection"
EMBEDDING_MODEL = "nomic-embed-text"


def parse_file_to_text(file_path: str) -> str:
    """Extracts raw string content from either a .pdf or .txt file."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content + "\n"
        return raw_text

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported format: {ext}")


def run_local_ingestion(file_path: str) -> int:
    """
    Purges old candidate records, chunks the new document, 
    embeds it via local GPU, and commits to disk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file {file_path} missing.")

    print(f"📄 Extracting raw text from: {file_path}")
    document_text = parse_file_to_text(file_path)

    # 1. Semantic Slice
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(document_text)

    # 2. Connect to Local Engine
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # 3. PRODUCTION PURGE: Wipe old resume out of memory
    existing_records = vector_store.get()
    if existing_records and existing_records["ids"]:
        print("🧹 Wiping previous candidate's vectors from disk...")
        vector_store.delete(ids=existing_records["ids"])

    # 4. Commit new candidate
    print(f"💾 Indexing {len(chunks)} new vector chunks...")
    vector_store.add_texts(texts=chunks)
    print("✅ Ingestion successfully locked to disk.")
    
    return len(chunks)

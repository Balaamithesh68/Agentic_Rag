import os
import json
import requests
import pypdf

API_URL = "http://localhost:8000/upload-resume"
SOURCE_DIR = "./test_resumes"
OUTPUT_JSONL = "training_data.jsonl"

def extract_raw_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def build_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"⚠️ Missing {SOURCE_DIR} folder.")
        return

    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".pdf")]
    print(f"🏭 Starting Synthetic Data Factory on {len(pdf_files)} files...")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
        for pdf_name in pdf_files:
            file_path = os.path.join(SOURCE_DIR, pdf_name)
            raw_text = extract_raw_text(file_path)
            
            # 1. Send to Phase 1 API to get verified Pydantic JSON
            with open(file_path, "rb") as f:
                files = {"file": (pdf_name, f, "application/pdf")}
                res = requests.post(API_URL, files=files)
                
            if res.status_code == 200:
                # In our Phase 1 API, trigger extraction or parse response
                # Here we format it as an OpenAI/HuggingFace instruction prompt
                ground_truth_json = res.json() 
                
                training_example = {
                    "instruction": "Extract the candidate profile into strict JSON.",
                    "input": raw_text,
                    "output": json.dumps(ground_truth_json)
                }
                
                outfile.write(json.dumps(training_example) + "\n")
                print(f"📦 Labeled and packed: {pdf_name}")
            else:
                print(f"❌ Failed to parse: {pdf_name}")

    print(f"✨ Training dataset compiled to {OUTPUT_JSONL}")

if __name__ == "__main__":
    build_dataset()

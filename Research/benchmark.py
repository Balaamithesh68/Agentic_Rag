import os
import time
import requests
import csv

# Target your running Phase 1 FastAPI Gateway
API_URL = "http://127.0.0.1:8000/upload-resume"
TEST_DIR = "./test_resumes"  # Put 3 to 5 sample PDF resumes in this folder
OUTPUT_CSV = "baseline_metrics.csv"

def run_benchmark():
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"⚠️ Created {TEST_DIR}. Please drop a few PDF resumes inside and re-run!")
        return

    pdf_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDFs found in {TEST_DIR}. Add some test resumes first.")
        return

    print(f"🏎️ Starting Benchmark Suite across {len(pdf_files)} documents...")
    
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["document_name", "file_size_kb", "latency_seconds", "status_code"])

        for pdf_name in pdf_files:
            file_path = os.path.join(TEST_DIR, pdf_name)
            file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
            
            print(f"⏳ Vectorizing & Parsing: {pdf_name} ({file_size_kb} KB)...")
            start_time = time.time()
            
            with open(file_path, "rb") as f:
                files = {"file": (pdf_name, f, "application/pdf")}
                response = requests.post(API_URL, files=files)
                
            latency = round(time.time() - start_time, 2)
            print(f"✅ Finished in {latency}s | Status: {response.status_code}")
            
            writer.writerow([pdf_name, file_size_kb, latency, response.status_code])

    print(f"📊 Benchmark complete! Baseline saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()

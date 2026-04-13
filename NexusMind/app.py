# app.py
import os
from flask import Flask, request, render_template, jsonify
from waitress import serve
from ingest import ingest_file
from embed_index import add_document, reset_index
from query_rag import answer_query
from query_rag import reset_history, format_citations
# Make sure to import the functions from hardware_check
from hardware_check import get_system_info, choose_preset
import psutil
import time
import torch
import subprocess

app = Flask(__name__, template_folder="templates", static_folder="static")
# Use a separate uploads directory to avoid collisions/permissions with repo-tracked files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Get system info and configuration on startup ---
SYSTEM_INFO = get_system_info()
PRESET, CONFIG = choose_preset(SYSTEM_INFO)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- API endpoint to provide system info to the UI ---
@app.route("/system_info", methods=["GET"])
def system_info():
    # Provide richer static info for header and status
    return jsonify({
        "preset": PRESET,
        "llm_model": os.path.basename(CONFIG.get("llm_model", "N/A")),
        "cpu_name": SYSTEM_INFO.get('cpu', 'N/A'),
        "cpu_max_ghz": SYSTEM_INFO.get('cpu_max_ghz'),
        "gpu_name": SYSTEM_INFO.get('gpu_name'),
        "gpu_vram_gb": SYSTEM_INFO.get('gpu_vram_gb')
    })

@app.route("/system_usage", methods=["GET"])
def system_usage():
    # Live usage percentages for UI polling
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_percent = psutil.virtual_memory().percent
    gpu_percent = None

    # Prefer NVML for accurate GPU utilization; fallback to nvidia-smi; last resort memory usage
    try:
        if torch.cuda.is_available():
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = float(util.gpu)
                pynvml.nvmlShutdown()
            except Exception:
                # Fallback to nvidia-smi CLI
                try:
                    smi_out = subprocess.check_output([
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu',
                        '--format=csv,noheader,nounits'
                    ], stderr=subprocess.STDOUT, text=True, timeout=1.5)
                    first_line = smi_out.strip().splitlines()[0].strip()
                    gpu_percent = float(first_line)
                except Exception:
                    # Last resort: approximate via allocated memory percentage
                    props = torch.cuda.get_device_properties(0)
                    total = props.total_memory
                    used = torch.cuda.memory_allocated(0)
                    gpu_percent = round((used / total) * 100.0, 1) if total else 0.0
    except Exception:
        gpu_percent = None

    return jsonify({
        "cpu": round(cpu_percent, 1),
        "ram": round(ram_percent, 1),
        "gpu": gpu_percent
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate a unique, safe filename
        base_name = os.path.basename(f.filename)
        name, ext = os.path.splitext(base_name)
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-","_"," ")).rstrip()
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_name = f"{safe_name or 'file'}_{ts}{ext}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        try:
            f.save(save_path)
        except PermissionError as pe:
            print(f"[ERROR] Permission denied saving to {save_path}: {pe}")
            return jsonify({"status": "error", "message": "Permission denied while saving file. Try a different file name or location."}), 500
        print(f"[UPLOAD] Saved file to: {save_path}")

        # Reset index and any chat/citation state for fresh context per upload
        reset_index()
        reset_history()
        
        # Ingest and chunk
        chunks, _ = ingest_file(save_path)
        print(f"[INGEST] Chunks detected: {len(chunks)}")
        
        doc_ids = []
        for text, metadata in chunks:
            if text and text.strip():
                doc_id = add_document(text, metadata)
                doc_ids.append(doc_id)
        print(f"[INDEX] Added chunks: {len(doc_ids)}")

        if doc_ids:
            return jsonify({"status": "ok", "doc_ids": doc_ids, "chunk_count": len(doc_ids)})
        else:
            return jsonify({"status": "empty", "message": "No text could be extracted from the file."})
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    q = request.form.get("query", "")
    if not q:
        return jsonify({"error": "no query"}), 400
    
    try:
        answer, results = answer_query(q, top_k=4) # Increased to 4 for better context
        citations = format_citations(results) if results else []
        return jsonify({"answer": answer, "results": results, "citations": citations})
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": "Query processing failed", "answer": "Sorry, an error occurred while processing your request."}), 500

if __name__ == "__main__":
    print(f"Starting NexusMind Server with '{PRESET}' profile...")
    print(f"Active LLM: {os.path.basename(CONFIG.get('llm_model', 'N/A'))}")
    
    # Fix Windows console issues by disabling Flask's banner
    import os
    os.environ['FLASK_SKIP_DOTENV'] = '1'
    
    # Use waitress for production deployment (more stable on Windows)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    serve(app, host='0.0.0.0', port=5000, threads=4)
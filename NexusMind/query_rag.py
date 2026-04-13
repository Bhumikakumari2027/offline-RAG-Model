# query_rag.py
import os
import re
from embed_index import search
from hardware_check import choose_preset
from llama_cpp import Llama

# --- DYNAMIC CONFIG LOADING (More Robust) ---
_, cfg = choose_preset()
# Use.get() for safety and use the correct key 'llm_model'
DEFAULT_MODEL = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" 
MODEL_PATH = cfg.get("llm_model", DEFAULT_MODEL)

# Instantiate LLM with dynamic configuration
print(f"Loading LLM: {os.path.basename(MODEL_PATH)}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=cfg.get("n_gpu_layers", -1),  # Use config from hardware_check
    verbose=False,
    temperature=0.1,
    top_p=0.9,
    top_k=40,
    repeat_penalty=1.18,
    stop=["<|eot_id|>", "<|end_of_text|>"]
)

PROMPT_TMPL = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a machine. Your task is to answer the user's question based ONLY on the provided sources.
- Answer concisely and factually.
- Cite the source number at the end of each sentence that uses information from a source, like this: [source:X].
- If the answer is not in the sources, state ONLY this: "I could not find an answer in the provided documents."

Sources:
---
{context}
---
<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def build_context_snippets(results):
    parts = []
    for i, r in enumerate(results, start=1):
        excerpt = r["text"].strip()
        source_path = r['metadata'].get('path', 'Unknown Source')
        
        # NEW: Check for audio metadata and add timestamp info
        if r['metadata'].get('type') == 'audio':
            start_time = r['metadata'].get('start_time', '00:00')
            end_time = r['metadata'].get('end_time', '00:00')
            source_info = f"Path: {source_path} (Time: {start_time} - {end_time})"
        else:
            source_info = f"Path: {source_path}"

        parts.append(f"Source {i}:\n{source_info}\nContent: {excerpt}")
    return "\n\n---\n\n".join(parts) 

# --- Simple in-memory chat state (optional) ---
CHAT_HISTORY = []

def reset_history():
    """Reset any in-memory chat or citation state used by the RAG pipeline.

    This is a safe no-op for now but keeps compatibility with callers in app.py.
    """
    global CHAT_HISTORY
    CHAT_HISTORY.clear()

def format_citations(results):
    """Create a lightweight citations structure for the UI from search results.

    Each entry includes a sequential source number and the original path. If the
    content originated from audio, include start/end timestamps when available.
    """
    citations = []
    for i, r in enumerate(results, start=1):
        metadata = r.get('metadata', {}) or {}
        citation = {
            'source': i,
            'path': metadata.get('path', 'Unknown Source')
        }
        if metadata.get('type') == 'audio':
            if 'start_time' in metadata:
                citation['start_time'] = metadata.get('start_time')
            if 'end_time' in metadata:
                citation['end_time'] = metadata.get('end_time')
        citations.append(citation)
    return citations

def answer_query(query, top_k=3):
    results = search(query, top_k=top_k)
    
    # Return both answer and an empty list for results to avoid errors in app.py
    if not results:
        return "I could not find any relevant documents to answer your question.",

    context = build_context_snippets(results)
    prompt = PROMPT_TMPL.format(context=context, question=query)
    
    resp = llm(prompt, max_tokens=256)
    
    # --- FINAL CORRECTED CODE ---

    # Step 1: Correctly access the text from the response structure
    # resp['choices'] is a LIST, so we take the first item 
    ans = resp["choices"][0]["text"].strip()

    # Step 2: Robustly clean up any lingering "assistant" tags
    # re.split returns a LIST, so we take the first item before stripping
    ans = re.split(r"assistant", ans, flags=re.IGNORECASE)[0].strip()
    
    return ans, results
# embed_index.py
import os, pickle, numpy as np
from sentence_transformers import SentenceTransformer
import threading
import faiss
from hardware_check import choose_preset

BASE_DIR = os.path.dirname(__file__)
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.bin")
META_FILE = os.path.join(BASE_DIR, "metadata.pkl")

# Global lock to prevent concurrent writes to index/metadata files
_index_io_lock = threading.Lock()

# load embedding model (this will download model first time if not present)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast
# Respect system preset for device selection
_, _cfg = choose_preset()
preferred_device = 'cuda' if _cfg.get('device') == 'cuda' else 'cpu'
try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=preferred_device)
except Exception as e:
    print(f"Preferred device '{preferred_device}' unavailable, falling back to CPU: {e}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
D = embed_model.get_sentence_embedding_dimension()

def init_index():
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        # ensure IDMap wrapper if not present
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
    else:
        flat = faiss.IndexFlatIP(D)
        index = faiss.IndexIDMap(flat) # to store ids

    # Check desired FAISS mode from preset and move index to GPU when requested
    try:
        if _cfg.get('faiss_mode') == 'gpu' and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()  # single GPU
            index = faiss.index_cpu_to_gpu(res, 0, index)
    except Exception as e:
        print(f"GPU FAISS not available, using CPU: {e}")

    return index

def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE,"rb") as f:
            meta = pickle.load(f)
    else:
        meta = []
    return meta

def save_metadata(meta):
    with open(META_FILE,"wb") as f:
        pickle.dump(meta,f)

def reset_index():
    """Clear FAISS index and metadata when a new document is uploaded."""
    with _index_io_lock:
        if os.path.exists(INDEX_FILE):
            try:
                os.remove(INDEX_FILE)
            except Exception as e:
                print(f"Failed to remove index file: {e}")
        if os.path.exists(META_FILE):
            try:
                os.remove(META_FILE)
            except Exception as e:
                print(f"Failed to remove metadata file: {e}")

def add_document(text, metadata):
    # Serialize operations that mutate index/metadata
    with _index_io_lock:
        index = init_index()
        meta = load_metadata()
        # id for this doc = len(meta)
        new_id = len(meta)
        emb = embed_model.encode([text], convert_to_numpy=True)
        # normalize for cosine similarity with IndexFlatIP
        faiss.normalize_L2(emb)
        index.add_with_ids(emb, np.array([new_id], dtype='int64'))
        meta.append({"id": new_id, "text": text, "metadata": metadata})
        save_metadata(meta)
        # Convert GPU index back to CPU before saving if needed
        try:
            if faiss.get_num_gpus() > 0:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, INDEX_FILE)
            else:
                faiss.write_index(index, INDEX_FILE)
        except Exception as e:
            print(f"Error saving index: {e}")
            # Fallback: try to save as CPU index
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, INDEX_FILE)
            except:
                faiss.write_index(index, INDEX_FILE)

        return new_id

def search(query_text, top_k=3):
    index = init_index()
    meta = load_metadata()
    if index.ntotal == 0:
        return []
    q_emb = embed_model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    Dists, Ids = index.search(q_emb, top_k)
    results = []
    for idx, dist in zip(Ids[0], Dists[0]):
        if idx == -1:
            continue
        entry = next((m for m in meta if m["id"]==int(idx)), None)
        if entry:
            results.append({"id":entry["id"], "score":float(dist), "text":entry["text"], "metadata":entry["metadata"]})
    return results

# index/retrieve.py
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer

# --- project-root absolute paths (safe) ---
SCRIPT_DIR = Path(__file__).resolve().parent      # index/
PROJECT_ROOT = SCRIPT_DIR.parent                  # mvp-copilot/
INDEX_DIR = (PROJECT_ROOT / "data" / "indexes").resolve()
CHUNKS_DIR = (PROJECT_ROOT / "data" / "chunks").resolve()

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_index():
    idx_path = INDEX_DIR / "vectors.index"
    meta_path = INDEX_DIR / "metadata.json"
    if not idx_path.exists():
        raise SystemExit(f"[!] vectors.index not found at {idx_path}. Run indexer first.")
    if not meta_path.exists():
        raise SystemExit(f"[!] metadata.json not found at {meta_path}. Run indexer first.")
    idx = faiss.read_index(str(idx_path))
    metadata = json.load(open(meta_path, "r", encoding="utf-8"))
    return idx, metadata

def query(q, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([q], convert_to_numpy=True)
    idx, metadata = load_index()
    D, I = idx.search(q_emb, top_k)
    results = []
    for i, score in zip(I[0], D[0]):
        meta = metadata[i]
        results.append({"score": float(score), "meta": meta})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print(f"[+] INDEX_DIR = {INDEX_DIR}")
    print(f"[+] CHUNKS_DIR = {CHUNKS_DIR}")
    res = query(args.q, top_k=args.k)
    for r in res:
        print("SCORE", r["score"], "FILE", r["meta"]["file_path"], r["meta"]["start_line"], r["meta"]["end_line"])
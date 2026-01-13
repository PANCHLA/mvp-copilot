from pathlib import Path
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# correct absolute project-root paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CHUNKS_DIR = (PROJECT_ROOT / "data" / "chunks").resolve()
INDEX_DIR = (PROJECT_ROOT / "data" / "indexes").resolve()

print("[+] CHUNKS_DIR =", CHUNKS_DIR)
print("[+] INDEX_DIR  =", INDEX_DIR)


MODEL_NAME = "BAAI/bge-small-en-v1.5"

def load_chunks(chunks_dir=CHUNKS_DIR):
    files = sorted([p for p in chunks_dir.glob("*.json")])
    docs = []
    for p in files:
        try:
            j = json.load(open(p, "r", encoding="utf-8"))
            docs.append((p.name, j))
        except Exception as e:
            print("skip", p, e)
    return docs

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def build_faiss_index(embeddings, dim):
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, metadata, out_dir=INDEX_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "vectors.index"))
    # metadata is list of dicts
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved index + metadata to {out_dir}")

if __name__ == "__main__":
    # load docs as before
    docs = load_chunks()
    if not docs:
        raise SystemExit("No chunks found. Run ingest first.")
    
    texts = []
    metadata = []
    for fname, j in docs:
        # compute a short relative path to include in embedding
        file_path = j.get("file_path", "")
        # Keep the file_path short: use last 4 path parts (or full if shorter)
        rel = Path(file_path).name if "/" not in file_path and "\\" not in file_path else file_path
        # Build embedding input: prepend the path + a separator
        embed_input = f"{rel}\n\n{j.get('text','')}"
        texts.append(embed_input)
        metadata.append({"id": fname, "file_path": file_path, "start_line": j.get("start_line"), "end_line": j.get("end_line")})

    print("[+] Loading embedder:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("[+] Computing embeddings (this can take some time)...")
    embeddings = embed_texts(texts, model)  # numpy array (N, D)
    dim = embeddings.shape[1]
    print(f"[+] Embeddings shape: {embeddings.shape}")

    print("[+] Building FAISS index...")
    index = build_faiss_index(embeddings, dim)
    save_index(index, metadata)

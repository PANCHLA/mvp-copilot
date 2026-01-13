# index/rag_orchestrator.py
"""
RAG orchestrator (retrieve -> prompt -> LLM)
Usage:
  python index/rag_orchestrator.py --q "explain auth" --k 5 --mode stub
Modes:
  - stub      : no API, returns simple synthesized answer (good for demos)
  - openrouter: requires OPENROUTER_API_KEY environment variable
  - local     : placeholder (implement local runtime call)
"""
from pathlib import Path
import json, os, textwrap, argparse
import uuid
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
import requests
import re
from math import ceil

# cache for cross-encoder
_cached_cross_encoder = None
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

# --- paths (project-root relative, absolute) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX_DIR = (PROJECT_ROOT / "data" / "indexes").resolve()
CHUNKS_DIR = (PROJECT_ROOT / "data" / "chunks").resolve()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Global variables to cache loaded resources
_cached_index = None
_cached_metadata = None
_chunk_text_cache = {}
_cached_model = None

# ---------------------------
# Index & Retrieval helpers
# ---------------------------
def get_model():
    global _cached_model
    if _cached_model is None:
        print("[+] Loading embedder...")
        _cached_model = SentenceTransformer(EMBED_MODEL)
    return _cached_model

def load_index_and_meta():
    global _cached_index, _cached_metadata
    if _cached_index is not None and _cached_metadata is not None:
        return _cached_index, _cached_metadata
        
    idx_path = INDEX_DIR / "vectors.index"
    meta_path = INDEX_DIR / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise SystemExit("[!] Missing index or metadata. Run indexer first.")
    _cached_index = faiss.read_index(str(idx_path))
    _cached_metadata = json.load(open(meta_path, "r", encoding="utf-8"))
    return _cached_index, _cached_metadata

def get_chunk_text(chunk_id):
    global _chunk_text_cache
    if chunk_id in _chunk_text_cache:
        return _chunk_text_cache[chunk_id]
    
    chunk_file = CHUNKS_DIR / chunk_id
    if chunk_file.exists():
        text = json.load(open(chunk_file, "r", encoding="utf-8")).get("text", "")
        _chunk_text_cache[chunk_id] = text
        return text
    return ""

def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    """
    Load and cache a CrossEncoder for re-ranking.
    """
    global _cached_cross_encoder
    if _cached_cross_encoder is None:
        try:
            print(f"[+] Loading cross-encoder: {model_name}")
            _cached_cross_encoder = CrossEncoder(model_name)
        except Exception as e:
            print(f"[WARN] Failed to load cross-encoder {model_name}: {e}")
            _cached_cross_encoder = None
    return _cached_cross_encoder


def rerank_with_cross_encoder(query: str, candidates: list, model=None, max_len: int = 1000):
    """
    Rerank candidate chunks using a cross-encoder.
    - `candidates` is a list of dicts with keys: 'meta', 'text', 'score' (as returned by retrieve_hybrid)
    - Returns the same list reordered and with 'rerank_score' attached to each candidate.
    """
    if not candidates:
        return candidates

    if model is None:
        model = get_cross_encoder()
    if model is None:
        # cannot rerank if model failed to load; return original order
        return candidates

    # Prepare pairs for cross-encoder: (query, chunk_text)
    pairs = []
    for c in candidates:
        text = c.get("text", "") or ""
        # Shorten chunk text to avoid model slowdown; prefix file path for stronger signal
        meta = c.get("meta", {})
        fp = meta.get("file_path", "")
        combined = f"{fp}\n\n{text}"
        if len(combined) > max_len:
            combined = combined[:max_len]
        pairs.append((query, combined))

    # Score all pairs
    try:
        scores = model.predict(pairs, batch_size=16)
    except Exception as e:
        print(f"[WARN] Cross-encoder scoring failed: {e}")
        return candidates

    # Attach rerank scores and sort (higher score = better)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates_sorted = sorted(candidates, key=lambda x: -x.get("rerank_score", 0.0))
    return candidates_sorted

def summarize_chunk(text: str, max_chars: int = 800, use_embed: bool = False, n_sentences: int = 3):
    """
    Lightweight extractive summarizer.
    - splits text into sentences,
    - if embeddings available, picks top n_sentences closest to the chunk centroid,
    - otherwise uses simple heuristics (prefer lines with keywords, else first N chars).
    """
    if not text:
        return ""

    # quick cleanup: remove long copyright headers or repeated license blocks
    text = re.sub(r"(?s)\*+ Copyright.*?$", "", text, flags=re.IGNORECASE)

    # split into sentences (simple)
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
    if not sentences:
        return text[:max_chars]

    keywords = {"auth","security","login","oauth","jwt","password","principal","filter","authorize","authentication","role","user"}
    # prefer sentences that contain keywords
    ranked = []
    if use_embed:
        try:
            # use the global get_model() if available
            model = get_model()
            # embed sentences and centroid
            s_embs = model.encode(sentences, convert_to_numpy=True)
            centroid = s_embs.mean(axis=0, keepdims=True)
            # compute cosine-ish similarity via dot (embeds not normalized but ranking works)
            sims = (s_embs @ centroid.T).flatten()
            ranked = sorted([(sims[i], sentences[i]) for i in range(len(sentences))], key=lambda x: -x[0])
            top = [s for _, s in ranked[:n_sentences]]
            summary = " ".join(top)
            if len(summary) > max_chars:
                return summary[:max_chars] + " ... [truncated]"
            return summary
        except Exception:
            # fallback to keyword heuristic below
            pass

    # Heuristic fallback: pick sentences with keywords
    key_sents = [s for s in sentences if any(k in s.lower() for k in keywords)]
    if key_sents:
        summary = " ".join(key_sents[:n_sentences])
        return summary[:max_chars] + (" ...[truncated]" if len(summary) > max_chars else summary)

    # final fallback: return first N chars (prefer whole sentences)
    accumulated = ""
    for s in sentences:
        if len(accumulated) + len(s) + 1 > max_chars:
            break
        accumulated += (s + " ")
    accumulated = accumulated.strip()
    if not accumulated:
        accumulated = text[:max_chars]
    return accumulated + (" ...[truncated]" if len(accumulated) >= max_chars else "")

def embed_query(q, model):
    """This function was missing, causing the NameError."""
    return model.encode([q], convert_to_numpy=True)

# ---------------------------
# Output sanitizer (top-level helper)
# ---------------------------
# What: Post-process answer to remove secrets/harmful tokens and detect hallucination beyond provided sources.
import re
SECRET_PATTERNS = [
    r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"(?i)password\s*[:=]\s*.+",
    r"(?i)-----BEGIN PRIVATE KEY-----",
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",   # IPs
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
]

def sanitize_output(text: str) -> str:
    # redact secrets
    for p in SECRET_PATTERNS:
        text = re.sub(p, "[REDACTED]", text)
    # remove suspicious external links
    text = re.sub(r"https?://\S+", "[LINK REDACTED]", text)
    # enforce instruction: don't invent files not in prompt
    # simple heuristic: if answer contains file paths not in sources, mark it
    return text


def retrieve_hybrid(q, top_k=5, model=None):
    if model is None:
        model = get_model()
    
    # Vector search
    q_emb = embed_query(q, model)
    idx, meta = load_index_and_meta()
    D, I = idx.search(q_emb, top_k)
    
    results = []
    for i, dist in zip(I[0], D[0]):
        m = meta[i]
        results.append({
            "score": float(dist),
            "meta": m,
            "text": get_chunk_text(m.get("id"))
        })
    
    # --- FALLBACK LOGIC ---
    # If the best result is a poor match, try a keyword search on file paths
    if results and results[0]["score"] > 35.0:  # A high score means low similarity
        print(f"[INFO] Vector search results were poor (score={results[0]['score']:.2f}). Trying keyword search on file paths.")
        keywords = q.lower().replace("?", "").split()
        keyword_results = []
        for i, m in enumerate(meta):
            file_path = m.get("file_path", "").lower()
            # Count keyword matches in the file path
            keyword_score = sum(1 for keyword in keywords if keyword in file_path)
            if keyword_score > 0:
                keyword_results.append({
                    "score": -keyword_score,  # Use a negative score so lower is better
                    "meta": m,
                    "text": get_chunk_text(m.get("id"))
                })
        
        # If keyword search found anything, use those results instead
        if keyword_results:
            keyword_results.sort(key=lambda x: x["score"])
            print(f"[INFO] Found {len(keyword_results)} results via keyword search on file paths.")
            return keyword_results[:top_k]
    
    return results

# ---------------------------
# Prompt template
# ---------------------------
PROMPT_HEADER = """You are a private code assistant. Use ONLY the retrieved code chunks below to answer the user's question. 
If the chunks don't contain direct information about the question, look for related concepts or patterns that might help answer the question.
ALWAYS cite the source as file_path:start_line-end_line for any code you reference. 
If you cannot answer from the context, explain what information would be needed to answer the question.

CRITICAL INSTRUCTION: Do not mention or infer the existence of any files, functions, or concepts that are not explicitly shown in the retrieved chunks above. Your entire answer must be based only on the text provided."""

PROMPT_FOOTER = "\n\nAnswer concisely and include references inline. If the retrieved chunks don't directly address the question, suggest what specific files or components might contain the answer."

def build_prompt(question, retrieved_chunks):
    parts = [PROMPT_HEADER, f"\n\nUser question: {question}\n\nRetrieved chunks (most relevant first):\n"]
    for idx, r in enumerate(retrieved_chunks, 1):
        m = r["meta"]
        fp = m.get("file_path", "<unknown>")
        sl = m.get("start_line", "?")
        el = m.get("end_line", "?")
        header = f"--- CHUNK {idx}: {fp}:{sl}-{el} ---\n"
        parts.append(header)
        # Text is already summarized/prepared by query_rag
        parts.append(r["text"] + "\n\n")
    parts.append(PROMPT_FOOTER)
    return "\n".join(parts)

# ---------------------------
# LLM adapters
# ---------------------------
def call_llm_stub(prompt):
    """Simple deterministic fallback: echo summary of retrieved context + simple heuristic."""
    lines = []
    for line in prompt.splitlines():
        if any(token in line.lower() for token in ["auth", "security", "application-", "@controller", "login", "password", "oauth", "jwt", "principal"]):
            lines.append(line.strip())
    snippet = "\n".join(lines[:10]).strip()
    if not snippet:
        snippet = "I cannot find explicit authentication logic in the provided chunks. See the cited files for details."
    answer = "Short summary (stubbed):\n" + snippet + "\n\n(References are shown in the retrieved chunks headers.)"
    return answer

def call_llm_openrouter(prompt):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("[!] OPENROUTER_API_KEY not set in environment for openrouter mode.")
    
    # Corrected URL based on your feedback
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {"role": "system", "content": "You are a helpful code assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.0
    }
    
    try:
        print("[+] Sending request to OpenRouter...")
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        print(f"[+] OpenRouter response status: {r.status_code}")
        
        if "choices" not in data or not data["choices"]:
            print("[!] Unexpected response structure from OpenRouter:")
            print(json.dumps(data, indent=2))
            return "Error: Unexpected response from OpenRouter API"
            
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"[!] Request to OpenRouter failed: {e}")
        return f"Error: Failed to connect to OpenRouter API: {e}"
    except Exception as e:
        print(f"[!] Error processing OpenRouter response: {e}")
        return f"Error: Failed to process OpenRouter response: {e}"

def call_llm_local(prompt):
    """
    Robust Ollama caller: handles streaming JSON lines and mixed bytes/str.
    """
    import requests, os, socket, json

    # quick reachability check (fast fail)
    try:
        sock = socket.create_connection(("127.0.0.1", 11434), timeout=0.8)
        sock.close()
    except Exception:
        return "[FALLBACK: Ollama not reachable quickly]\n" + call_llm_stub(prompt)

    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2:1b"),
        "prompt": prompt,
        "max_tokens": 150,
        "stream": True
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=(3, 180)) as r:
            r.raise_for_status()
            pieces = []

            # iterate lines (bytes). Use decode manually to robustly handle both bytes/str.
            for raw in r.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                # raw may be bytes or str; normalize to str
                line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                line = line.strip()
                if not line:
                    continue

                # Strip possible leading request-id before JSON: find first '{'
                if line and line[0] != "{":
                    idx = line.find("{")
                    if idx != -1:
                        line = line[idx:]
                # Parse
                try:
                    obj = json.loads(line)
                except Exception:
                    # not JSON: append raw text defensively
                    pieces.append(line)
                    continue

                # Extract common fields
                if isinstance(obj, dict):
                    # direct token stream field
                    if obj.get("response"):
                        pieces.append(str(obj.get("response") or ""))
                        continue
                    # newer shape: outputs -> content -> output_text
                    outs = obj.get("outputs") or obj.get("result") or []
                    if isinstance(outs, list) and outs:
                        extracted = False
                        for out in outs:
                            if isinstance(out, dict):
                                # older simple text
                                if out.get("text"):
                                    pieces.append(str(out.get("text")))
                                    extracted = True
                                    break
                                # content blocks
                                cont = out.get("content") or []
                                if isinstance(cont, list):
                                    for c in cont:
                                        if isinstance(c, dict) and c.get("type") == "output_text" and c.get("text"):
                                            pieces.append(str(c.get("text")))
                                            extracted = True
                                            break
                                    if extracted:
                                        break
                        if extracted:
                            continue
                    # fallback: if obj has 'response' empty but other useful keys, stringify
                    # append nothing here and keep looping
                else:
                    pieces.append(str(obj))

            final = "".join(pieces).strip()
            if final:
                return final
            # fallback: try full response body
            try:
                full = r.text
                return full[:4000] if full else call_llm_stub(prompt)
            except Exception:
                return call_llm_stub(prompt)

    except requests.exceptions.RequestException as e:
        return f"[FALLBACK: Ollama error: {e}]\n\n" + call_llm_stub(prompt)
    except Exception as e:
        return f"[FALLBACK: Ollama unexpected error: {e}]\n\n" + call_llm_stub(prompt)


def query_rag(
    query_text: str,
    top_k: int = 5,
    mode: str = "stub",
    summarize: bool = True,
    summarize_chars: int = 700,
    summary_sentences: int = 3,
    re_rank: bool = False,
    context: str = ""
):
    """
    Clean, efficient RAG entrypoint with timing instrumentation.
    """

    import time
    t0 = time.time()
    print("\n===== RAG PIPELINE START =====")

    try:
        # ---------------------------------------
        # 1) Load embedder (cached)
        # ---------------------------------------
        t_embedder_start = time.time()
        model = get_model()
        print(f"[TIMING] embedder ready in {time.time() - t_embedder_start:.3f}s")


        # ---------------------------------------
        # 2) Retrieval
        # ---------------------------------------
        t_retr_start = time.time()
        retrieved = retrieve_hybrid(query_text, top_k=top_k, model=model) or []
        print(f"[TIMING] retrieval = {time.time() - t_retr_start:.3f}s, results = {len(retrieved)}")

        if not retrieved:
            msg = "No relevant code chunks were found in the index for your query."
            print("[INFO] Empty retrieval")
            return {"answer": msg, "sources": [], "prompt": ""}


        # ---------------------------------------
        # 3) Normalize retrieved chunks
        # ---------------------------------------
        normalized = []
        for r in retrieved:
            meta = r.get("meta", {})
            chunk_id = meta.get("id")
            orig_text = r.get("text") or get_chunk_text(chunk_id) or ""
            normalized.append({
                "file": meta.get("file_path"),
                "start": meta.get("start_line"),
                "end": meta.get("end_line"),
                "score": r.get("score"),
                "id": chunk_id,
                "text": orig_text,
                "summary": ""
            })

        # ---------------------------------------
        # 3.5) Inject Context (Active File)
        # ---------------------------------------
        if context:
            print("[INFO] Injecting active file context into prompt.")
            # Create a synthetic chunk for the active file
            # We put it at the START of the list so it's prioritized
            normalized.insert(0, {
                "file": "ACTIVE_EDITOR_CONTEXT",
                "start": 0,
                "end": 0,
                "score": 999.9, # Artificial high score
                "id": "context",
                "text": context,
                "summary": "" # Will be filled by summarizer if needed, or we can skip
            })

        # ---------------------------------------
        # 4) Optional Re-ranking (placeholder)
        # ---------------------------------------
        if re_rank:
            t_rerank_start = time.time()
            # raw retrieved (retrieve_hybrid output) is compatible with reranker
            # rerank_with_cross_encoder expects list of dicts with 'meta' and 'text'
            try:
                # run the re-ranker and reorder 'retrieved' (we keep the same item shape)
                retrieved = rerank_with_cross_encoder(query_text, retrieved[:3], model=None, max_len=1200)
            except Exception as e:
                print(f"[WARN] re-rank failed: {e}")
            print(f"[TIMING] re_rank = {time.time() - t_rerank_start:.3f}s")

        # ---------------------------------------
        # 5) Summarization step
        # ---------------------------------------
        t_sum_start = time.time()
        if summarize:
            for e in normalized:
                # >>> ADD THIS BLOCK <<<
                if e["id"] == "context":
                    e["summary"] = e["text"]
                    continue
                # >>> END BLOCK <<<
                try:
                    e["summary"] = summarize_chunk(
                        e["text"],
                        max_chars=summarize_chars,
                        use_embed=False,        # fast heuristic summarizer
                        n_sentences=summary_sentences
                    )
                except Exception:
                    e["summary"] = (e["text"] or "")[:min(400, summarize_chars)]
        print(f"[TIMING] summarization = {time.time() - t_sum_start:.3f}s")

        # ---------------------------------------
        # 6) Build prompt
        # ---------------------------------------
        t_prompt_start = time.time()
        prompt_chunks = []
        for e in normalized:
            prompt_chunks.append({
                "meta": {
                    "file_path": e["file"],
                    "start_line": e["start"],
                    "end_line": e["end"],
                    "id": e["id"]
                },
                "text": e["summary"] or (e["text"][:summarize_chars] if e["text"] else "")
            })

        prompt = build_prompt(query_text, prompt_chunks)
        print(f"[TIMING] prompt_build = {time.time() - t_prompt_start:.3f}s")

        # ---------------------------------------
        # 7) LLM call
        # ---------------------------------------
        t_llm_start = time.time()
        if mode == "openrouter":
            answer = call_llm_openrouter(prompt)
        elif mode == "local":
            answer = call_llm_local(prompt)
        else:
            answer = call_llm_stub(prompt)
        print(f"[TIMING] llm = {time.time() - t_llm_start:.3f}s")

        # >>> ADD THIS <<<
        answer = sanitize_output(answer)

        

        # ---------------------------------------
        # 9) Prompt & Retrieval Logging audit (add near end of query_rag, before return)
        # ---------------------------------------
        import uuid, time, json
        AUDIT_LOG = PROJECT_ROOT / "data" / "audit.log"

        entry = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "query": query_text,
            "mode": mode,
            "sources": [{"file": e["file"], "start": e["start"], "end": e["end"], "score": e["score"]} for e in normalized],
            "prompt": prompt,
            "answer": answer[:8000],   # truncate very long outputs
            "note": "local" if mode=="local" else "openrouter"
        }
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


        # ---------------------------------------
        # 10) Build response
        # ---------------------------------------
        sources = []
        for e in normalized:
            sources.append({
                "file": e["file"],
                "start": e["start"],
                "end": e["end"],
                "score": e["score"],
                "id": e["id"],
                "text": e["text"],
                "summary": e["summary"]
            })

        print(f"[TIMING] TOTAL pipeline = {time.time() - t0:.3f}s")
        print("===== RAG PIPELINE END =====\n")

        return {"answer": answer, "sources": sources, "prompt": prompt}

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()

        fallback = call_llm_stub(
            f"RAG ERROR: {exc}\nOriginal query: {query_text}"
        )

        print("[ERROR] Pipeline crashed:", exc)
        print(tb)

        return {"answer": fallback, "sources": [], "prompt": ""}


# ---------------------------
# Main CLI
# ---------------------------
# In index/rag_orchestrator.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["stub", "openrouter", "local"], default="stub")
    args = parser.parse_args()

    try:
        print("[+] Loading embedder...")
        # This is just to pre-load the model and see the message
        get_model()
        
        print("[+] Calling query_rag...")
        # Call the high-level function which does everything
        result = query_rag(args.q, top_k=args.k, mode=args.mode, summarize=True, summarize_chars=700)
        
        # Print the sources for visibility
        print("\n--- Retrieved Sources ---")
        for src in result["sources"]:
            print(f"  - {src['file']}:{src['start']}-{src['end']} (score={src['score']:.4f})")

        print("\n=== LLM ANSWER ===\n")
        print(result["answer"])
        print("\n=== END ===")
    except Exception as e:
        print(f"[!] Error in RAG orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
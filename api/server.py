from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import sys
from pathlib import Path
import os
from fastapi import Depends


# Add project root to sys.path so we can import from index
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from index.rag_orchestrator import query_rag, get_model, load_index_and_meta

app = FastAPI(title="MVP Copilot RAG API")

class QueryRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "stub"
    context: str = ""

@app.on_event("startup")
async def startup_event():
    print("[API] Pre-loading RAG resources...")
    try:
        get_model()
        load_index_and_meta()
        print("[API] RAG resources loaded.")
    except Exception as e:
        print(f"[API] Failed to load RAG resources: {e}")

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        result = query_rag(request.q, top_k=request.k, mode=request.mode, context=request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# --- Admin audit endpoint & auth (fixed) ---
import json
from fastapi import Header, HTTPException

AUDIT_LOG = PROJECT_ROOT / "data" / "audit.log"

def admin_auth(x_api_key: str = Header(None)):
    if x_api_key != os.environ.get("ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

@app.get("/admin/audit")
def get_audit(n: int = 50, _: str = Depends(admin_auth)):
    # ensure audit file exists
    try:
        if not AUDIT_LOG.exists():
            return {"count": 0, "last": []}
        out = []
        with open(AUDIT_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except:
                    continue
        return {"count": len(out), "last": out[-n:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

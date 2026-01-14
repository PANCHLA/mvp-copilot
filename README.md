# MVP Copilot

**A privacy-first, local RAG code assistant for VS Code.**

Runs locally to keep your code private. "Chat with your codebase" without data leaving your machine.

---

##  Architecture

- **Frontend**: VS Code Extension (TypeScript)
- **Backend**: FastAPI Server (Python)
- **RAG Engine**: FAISS (Vector Store) + BAAI Embeddings + Cross-Encoder Re-ranking

---

##  What I Have Built

- **Hybrid RAG Pipeline**: Combines vector search, keyword fallback, and cross-encoder re-ranking for high precision.
- **Smart Context**: Automatically injects your active file and folder summaries into the prompt.
- **Privacy & Security**: Runs offline (Ollama) and auto-redacts secrets (API keys, passwords) from outputs.
- **Flexible Models**: Switch between Local (Llama 3, Mistral) and Cloud (OpenRouter/GPT-4).
- **Audit Logging**: Full transparency log of every query, source, and answer.

---

##  What's Missing 

- **Real-Time Indexing**: Index is static; requires manual re-run to capture code changes.
- **Multi-Turn Conversation**: Currently supports single-turn QA only.
- **Agentic Capabilities**: Read-only system; cannot write to files or execute commands.
- **Deep Code Graph**: No dependency or call-graph analysis yet.
- **Production Auth**: Basic API key check only.

---

## 🚀 Quick Setup & Usage

### 1. Install & Ingest
```bash
pip install -r requirements.txt
python ingest/ingest.py --git https://github.com/username/repo.git --name my_repo
```

### 2. Chat via CLI (No UI)
```bash
# Ask questions directly in terminal
python index/rag_orchestrator.py --q "How does login work?" --mode local
```

### 3. Run Full App
```bash
# Start backend API
python api/server.py

# Then open `mvp-extension/` in VS Code and press F5 to use the Chat UI.
```

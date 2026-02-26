from pathlib import Path

# ── Ollama endpoints ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.2-4k"       # created with: ollama create llama3.2-4k -f Modelfile
EMBED_MODEL     = "nomic-embed-text"

# ── Context window ────────────────────────────────────────────────────────────
# llama3.2 defaults to 128k tokens (~14 GB KV cache). 4096 is sufficient for RAG.
NUM_CTX         = 4096

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K           = 3

# ── Paths (relative to project root) ─────────────────────────────────────────
_ROOT    = Path(__file__).parent.parent
DATA_DIR    = _ROOT / "data"
PERSIST_DIR = _ROOT / "storage"

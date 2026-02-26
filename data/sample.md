# Retrieval-Augmented Generation (RAG) — Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI architecture pattern that improves the accuracy and relevance of language model outputs by retrieving supporting documents **before** generating a response. Rather than forcing a model to recall everything from its training weights, RAG lets the model "look up" relevant facts on demand.

---

## Core Benefits of RAG

| Benefit | Description |
|---|---|
| **Reduced hallucinations** | Answers are grounded in retrieved documents, not invented from memory. |
| **Fresh knowledge** | Works with documents updated after the model's training cutoff. |
| **Domain specificity** | Handles proprietary, confidential, or niche knowledge bases. |
| **Transparency** | Retrieved source chunks can be shown to users for auditability. |
| **Cost efficiency** | Smaller local models (e.g., llama3.2 via Ollama) can perform well with good context. |

---

## RAG Pipeline Steps

### Step 1 — Document Ingestion
Load raw documents from files (PDF, TXT, Markdown, HTML) or external sources. Tools like `SimpleDirectoryReader` in LlamaIndex automate this step.

### Step 2 — Chunking
Split documents into smaller, overlapping segments (nodes). Smaller chunks improve retrieval precision; overlap prevents context loss at boundaries.

```
Document → [Chunk 1] [Chunk 2] [Chunk 3] ...
                  ↕ overlap ↕ overlap
```

Common parameters:
- `chunk_size`: number of tokens/characters per chunk (e.g., 512)
- `chunk_overlap`: tokens shared between consecutive chunks (e.g., 50)

### Step 3 — Embedding
Each chunk is converted into a dense numeric vector using an embedding model. Semantically similar chunks have vectors that are close together in the embedding space.

Recommended embedding models via Ollama:
- `nomic-embed-text` — purpose-built for retrieval, fast and accurate
- `llama3.2` — usable as a fallback but slower

### Step 4 — Indexing
Embeddings are stored in a vector index (e.g., `VectorStoreIndex`). The index supports fast approximate nearest-neighbour search.

### Step 5 — Querying
At query time:
1. Embed the user's question with the **same** embedding model.
2. Search the index for the top-K most similar chunks (`similarity_top_k`).
3. Pass the retrieved chunks as context to the LLM.
4. The LLM synthesises a final answer.

### Step 6 — Response Synthesis
LlamaIndex offers several response modes:
- `compact` — fits as many retrieved chunks into one prompt as possible (default, efficient)
- `refine` — iteratively refines the answer over each chunk (thorough but slower)
- `tree_summarize` — builds a summary tree (good for large result sets)

---

## Persistence

After building a vector index, persist it to disk so you don't re-embed on every run:

```python
index.storage_context.persist(persist_dir="./storage")
```

Load it back later:

```python
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

---

## When to Use RAG vs Fine-Tuning

| Scenario | RAG | Fine-Tuning |
|---|---|---|
| Frequently changing data | ✅ Ideal | ❌ Requires retraining |
| Proprietary documents | ✅ Ideal | ⚠️ Data exposure risk |
| Style/behaviour changes | ❌ Limited | ✅ Ideal |
| Low-latency requirements | ⚠️ Adds retrieval step | ✅ No retrieval needed |
| Auditability / citations | ✅ Built-in | ❌ Opaque |

---

## Quick-Start Checklist

- [ ] Install Ollama and start the server: `ollama serve`
- [ ] Pull required models: `ollama pull llama3.2 && ollama pull nomic-embed-text`
- [ ] Sync the uv project environment: `uv sync`
- [ ] Place documents in the `./data/` directory
- [ ] Run the notebook cells top-to-bottom
- [ ] Edit the `QUESTION` variable in Section 7 to ask your own questions

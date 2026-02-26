"""
RAG pipeline helpers: settings init, index build/load, engine factories.
"""

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    NUM_CTX,
    OLLAMA_BASE_URL,
    PERSIST_DIR,
    TOP_K,
)


def init_settings() -> tuple:
    """Instantiate LLM + embedding model and register on Settings singleton."""
    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0,
        temperature=0.1,
        context_window=NUM_CTX,
    )
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def build_or_load_index() -> VectorStoreIndex:
    """
    Load the persisted index from disk if available,
    otherwise build from documents in DATA_DIR and persist.
    """
    init_settings()

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        print(f"Loading index from '{PERSIST_DIR}'…")
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_context)

    print(f"Building index from '{DATA_DIR}'…")
    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
        required_exts=[".txt", ".md", ".pdf"],
    )
    documents = reader.load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    print(f"Index saved to '{PERSIST_DIR}'.")
    return index


def get_query_engine(index: VectorStoreIndex):
    """Single-turn query engine."""
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
    )


def get_chat_engine(index: VectorStoreIndex):
    """
    Multi-round chat engine using condense_plus_context mode:
    condenses conversation history into a standalone question,
    then retrieves relevant chunks for grounded answers.
    """
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=TOP_K,
        verbose=False,
    )

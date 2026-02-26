"""
Streamlit multi-round Q&A frontend for the Epstein Files RAG pipeline.

Run with:
    uv run streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from rag import build_or_load_index, get_chat_engine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epstein Files Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Epstein Files Q&A")
st.caption(
    "Multi-round chat grounded in the DOJ Epstein disclosure documents. "
    "Ask follow-up questions — the assistant remembers the conversation."
)

# ── Load index + chat engine (cached across reruns) ───────────────────────────
@st.cache_resource(show_spinner="Loading document index…")
def load_engine():
    index = build_or_load_index()
    return get_chat_engine(index)


chat_engine = load_engine()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        chat_engine.reset()
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Documents in `./data` are chunked and embedded.\n"
        "2. Your question is matched against the vector index.\n"
        "3. The top-3 chunks are passed to `llama3.2` as context.\n"
        "4. Follow-up questions are condensed with chat history "
        "before retrieval."
    )

# ── Render existing chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"**{src['file']}** — score: `{src['score']:.3f}`")
                    st.caption(src["excerpt"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the Epstein files…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = chat_engine.chat(prompt)

        answer = str(response)
        st.markdown(answer)

        # Source attribution
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for node in response.source_nodes:
                sources.append({
                    "file":    node.metadata.get("file_name", "unknown"),
                    "score":   node.score if node.score is not None else 0.0,
                    "excerpt": node.get_content()[:300].replace("\n", " "),
                })
            with st.expander("Sources", expanded=False):
                for src in sources:
                    st.markdown(f"**{src['file']}** — score: `{src['score']:.3f}`")
                    st.caption(src["excerpt"])

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

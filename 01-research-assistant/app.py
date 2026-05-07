"""Streamlit UI for Paperback — citation-grounded RAG over ML papers."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import chromadb
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from src.generate import generate_answer
from src.ingest import ingest_pdf
from src.retrieve import retrieve

load_dotenv()

st.set_page_config(page_title="Paperback", page_icon="📚", layout="wide")
st.title("Paperback")
st.caption("Citation-grounded research assistant for ML papers.")


# --- clients (cached) and collection (always fresh) ---
@st.cache_resource
def get_clients():
    """API clients only. Cached, since they're stateless."""
    openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not openai_key or not anthropic_key:
        st.error("Missing API keys. Set OPENAI_API_KEY and ANTHROPIC_API_KEY in .env or Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=openai_key), Anthropic(api_key=anthropic_key)


def get_collection():
    """Fetch (or create) the Chroma collection. Not cached — collections can be deleted."""
    chroma = chromadb.PersistentClient(path="./chroma_db")
    return chroma.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})


openai_client, anthropic_client = get_clients()
collection = get_collection()


# --- sidebar: ingest ---
with st.sidebar:
    st.header("Corpus")
    st.write(f"**{collection.count()}** chunks indexed")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Index uploaded papers", type="primary"):
        with st.status("Indexing...", expanded=True) as status:
            for f in uploaded:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = Path(tmp.name)
                summary = ingest_pdf(tmp_path, collection, openai_client)
                tmp_path.unlink(missing_ok=True)
                st.write(f"  • {summary['title']} — {summary['chunks']} chunks")
            status.update(label="Done.", state="complete")
        st.rerun()

    if st.button("Clear corpus", type="secondary"):
        chroma = chromadb.PersistentClient(path="./chroma_db")
        try:
            chroma.delete_collection("papers")
        except Exception:
            pass  # already gone
        st.rerun()  # next render will recreate via get_collection()


# --- main: ask ---
st.subheader("Ask a question")

question = st.text_input(
    "Question",
    placeholder="e.g., How does the paper handle long-context retrieval?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    k = st.number_input("Top-k", 3, 12, 6, help="Number of chunks to retrieve. PRD recommends 6.")

if question:
    if collection.count() == 0:
        st.warning("Upload some papers first.")
        st.stop()

    with st.spinner("Retrieving..."):
        chunks = retrieve(question, collection, openai_client, k=int(k))

    if not chunks:
        st.error("No relevant chunks found.")
        st.stop()

    with st.spinner("Generating answer..."):
        answer = generate_answer(question, chunks, anthropic_client)

    st.markdown("### Answer")
    st.markdown(answer)

    st.markdown("### Sources")
    st.caption("Shown by default. The whole point is verifiability.")
    for i, c in enumerate(chunks, 1):
        with st.expander(f"{i}. {c.citation_tag()} — {c.section} (cosine distance: {c.distance:.3f})"):
            st.write(c.text)

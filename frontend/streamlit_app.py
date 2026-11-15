"""Streamlit frontend: upload healthcare documents and chat with them.

Run:  streamlit run frontend/streamlit_app.py     (the backend must be running)
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
FILE_TYPES = ["pdf", "docx", "txt", "md", "csv"]

st.set_page_config(page_title="Healthcare RAG Assistant", page_icon="🩺", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call the backend; return (data, error message)."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=kwargs.pop("timeout", 300), **kwargs)
    except requests.RequestException:
        return None, f"Cannot reach the backend at {BACKEND_URL}. Start it with: uvicorn app.backend.main:app"
    if resp.ok:
        return resp.json(), None
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return None, f"Error {resp.status_code}: {detail}"


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = f" · page {s['page']}" if s.get("page") else ""
            used = " · cited" if s.get("cited") else ""
            st.markdown(f"**[{s['index']}] {s['file_name']}**{page} · relevance {s['score']:.2f}{used}")
            st.caption(s["section"])
            st.write(s["text"])
            st.divider()


# ------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload healthcare documents", type=FILE_TYPES, accept_multiple_files=True,
                               help="PDF, DOCX, TXT, MD or CSV")
    if st.button("Add to knowledge base", disabled=not uploads, type="primary", use_container_width=True):
        with st.spinner("Reading, chunking and indexing…"):
            files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
            results, err = api("POST", "/api/documents/upload", files=files, timeout=3600)
        if err:
            st.error(err)
        else:
            for r in results:
                if r["status"] == "ingested":
                    st.success(f"{r['file_name']}: {r['chunks']} chunks")
                else:
                    st.error(f"{r['file_name']}: {r['error']}")

    docs, err = api("GET", "/api/documents", timeout=30)
    if err:
        st.warning(err)
        docs = []
    st.subheader(f"Indexed documents ({len(docs)})")
    selected = []
    for d in docs:
        left, right = st.columns([5, 1])
        if left.checkbox(f"{d['file_name']} ({d['chunks']})", value=True, key=f"use_{d['doc_id']}"):
            selected.append(d["doc_id"])
        if right.button("✕", key=f"del_{d['doc_id']}", help="Remove this document"):
            api("DELETE", f"/api/documents/{d['doc_id']}")
            st.rerun()

    st.subheader("Search settings")
    top_k = st.slider("Passages sent to the LLM", 1, 10, 5)
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"], help="hybrid = BM25 + vector search (recommended)")
    use_reranker = st.toggle("Rerank results", value=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------- chat
st.title("🩺 Healthcare RAG Assistant")
st.caption("Answers come only from your uploaded documents, with sources. This is not medical advice.")

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not docs:
            answer, sources = "Please upload at least one document first.", []
        elif not selected:
            answer, sources = "Please tick at least one document in the sidebar.", []
        else:
            with st.spinner("Searching your documents…"):
                payload = {"question": question, "top_k": top_k, "mode": mode, "rerank": use_reranker,
                           "doc_ids": selected if len(selected) < len(docs) else None}
                data, err = api("POST", "/api/query", json=payload)
            answer, sources = (err, []) if err else (data["answer"], data["sources"])
        st.markdown(answer)
        show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

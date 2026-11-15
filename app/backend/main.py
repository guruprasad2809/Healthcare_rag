"""FastAPI backend: connects the frontend to ingestion, retrieval and the Gemini LLM.

Run:  uvicorn app.backend.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend import container
from app.backend.llm import LLMError
from app.backend.schemas import DocumentInfo, QueryRequest, QueryResponse, UploadResult
from app.config import get_settings
from app.ingestion.parsers import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("healthcare_rag")
settings = get_settings()

app = FastAPI(title="Healthcare RAG API", version="1.0.0",
              description="Upload healthcare documents (PDF, DOCX, TXT, MD, CSV) and ask questions answered from them.")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_model": settings.gemini_model, "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model, "gemini_key_set": bool(settings.gemini_api_key)}


@app.post("/api/documents/upload", response_model=list[UploadResult])
def upload_documents(files: list[UploadFile] = File(...), pipeline=Depends(container.get_ingestion)) -> list[UploadResult]:
    """Upload one or more files; each is parsed, chunked, embedded and indexed."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for file in files:
        name = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "upload").name)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append(UploadResult(file_name=name, status="failed", error=f"Unsupported file type '{ext}'"))
            continue
        data = file.file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            results.append(UploadResult(file_name=name, status="failed", error=f"File is larger than {settings.max_upload_mb} MB"))
            continue
        path = upload_dir / f"{uuid.uuid4().hex[:8]}_{name}"
        path.write_bytes(data)
        try:
            r = pipeline.ingest_file(path, name)
            results.append(UploadResult(file_name=name, status="ingested", doc_id=r["doc_id"], chunks=r["chunks"], seconds=r["seconds"]))
        except Exception as exc:
            log.exception("Ingestion failed for %s", name)
            path.unlink(missing_ok=True)
            results.append(UploadResult(file_name=name, status="failed", error=str(exc)))
    return results


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents(store=Depends(container.get_store)) -> list[dict]:
    return store.list_documents()


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, store=Depends(container.get_store)) -> dict:
    store.delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, rag=Depends(container.get_rag)) -> dict:
    """Hybrid search + rerank, then a Gemini answer grounded in the retrieved passages."""
    try:
        return rag.answer(req.question, top_k=req.top_k, doc_ids=req.doc_ids, mode=req.mode, rerank=req.rerank)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

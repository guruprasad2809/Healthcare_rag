"""Ingestion pipeline: file -> parse -> clean -> chunk -> embed -> Qdrant."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from app.ingestion.chunking import SemanticTitleChunker
from app.ingestion.embeddings import BM25SparseEncoder
from app.ingestion.parsers import parse_document
from app.retrieval.vector_store import QdrantStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, embedder, sparse_encoder: BM25SparseEncoder, store: QdrantStore, chunker: SemanticTitleChunker):
        self.embedder, self.sparse_encoder, self.store, self.chunker = embedder, sparse_encoder, store, chunker

    def ingest_file(self, path: str | Path, file_name: str | None = None) -> dict:
        """Ingest one file. Re-uploading the same file replaces its old chunks."""
        started = time.perf_counter()
        doc = parse_document(path, file_name)
        chunks = self.chunker.chunk(doc)
        texts = [c.embed_text for c in chunks]
        dense = self.embedder.embed_documents(texts)
        sparse = [self.sparse_encoder.encode_document(t) for t in texts]

        doc_payload = {"doc_id": doc.doc_id, "file_name": doc.file_name, "file_type": doc.file_type, "doc_title": doc.title,
                       "ingested_at": doc.metadata["ingested_at"]}
        self.store.delete_document(doc.doc_id)
        self.store.upsert_chunks(chunks, dense, sparse, doc_payload)
        result = {**doc_payload, "chunks": len(chunks), "seconds": round(time.perf_counter() - started, 2),
                  "metadata": {k: v for k, v in doc.metadata.items() if k != "ingested_at"}}
        log.info("Ingested %s: %d chunks in %.1fs", doc.file_name, len(chunks), result["seconds"])
        return result

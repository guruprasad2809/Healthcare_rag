"""Qdrant storage: one collection, two named vectors per chunk (dense + BM25 sparse)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Sequence

from qdrant_client import QdrantClient, models

from app.ingestion.chunking import Chunk

DENSE, SPARSE = "dense", "bm25"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    title_path: list[str]
    doc_id: str
    file_name: str
    page: int | None = None
    score: float = 0.0  # fusion score, replaced by the reranker score when reranked
    scores: dict = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        trail = " > ".join(self.title_path)
        return f"{trail}\n{self.text}" if trail else self.text


def make_client(url: str = "", api_key: str = "", local_path: str = "qdrant_storage") -> QdrantClient:
    """Remote Qdrant when a URL is configured, otherwise an embedded on-disk instance."""
    if url:
        return QdrantClient(url=url, api_key=api_key or None, timeout=60)
    return QdrantClient(path=local_path)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client, self.collection, self.dim = client, collection, dim
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for key in ("doc_id", "file_type"):  # fast filtering by document (no-op in embedded mode)
            self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)

    # ------------------------------------------------------------------ writes
    def upsert_chunks(self, chunks: Sequence[Chunk], dense: Sequence[Sequence[float]], sparse: Sequence[tuple[list[int], list[float]]],
                      doc_payload: dict, batch_size: int = 64) -> int:
        points = []
        for chunk, vec, (idx, val) in zip(chunks, dense, sparse):
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.doc_id}:{chunk.chunk_index}")),
                vector={DENSE: list(vec), SPARSE: models.SparseVector(indices=idx, values=val)},
                payload={**doc_payload, "text": chunk.text, "title_path": chunk.title_path, "section_title": chunk.section_title,
                         "chunk_index": chunk.chunk_index, "page": chunk.page, **chunk.metadata},
            ))
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[i:i + batch_size], wait=True)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, models.FilterSelector(filter=self._filter([doc_id])), wait=True)

    # ------------------------------------------------------------------- reads
    @staticmethod
    def _filter(doc_ids: Sequence[str] | None):
        if not doc_ids:
            return None
        return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))])

    @staticmethod
    def _to_chunk(point, score_name: str) -> RetrievedChunk:
        p = point.payload or {}
        return RetrievedChunk(str(point.id), p.get("text", ""), p.get("title_path", []), p.get("doc_id", ""), p.get("file_name", ""),
                              p.get("page"), float(point.score), {score_name: float(point.score)})

    def dense_search(self, vector: Sequence[float], limit: int, doc_ids: Sequence[str] | None = None) -> list[RetrievedChunk]:
        res = self.client.query_points(self.collection, query=list(vector), using=DENSE, limit=limit, query_filter=self._filter(doc_ids), with_payload=True)
        return [self._to_chunk(p, "dense") for p in res.points]

    def bm25_search(self, sparse: tuple[list[int], list[float]], limit: int, doc_ids: Sequence[str] | None = None) -> list[RetrievedChunk]:
        if not sparse[0]:
            return []
        res = self.client.query_points(self.collection, query=models.SparseVector(indices=sparse[0], values=sparse[1]), using=SPARSE,
                                       limit=limit, query_filter=self._filter(doc_ids), with_payload=True)
        return [self._to_chunk(p, "bm25") for p in res.points]

    def hybrid_search(self, vector: Sequence[float], sparse: tuple[list[int], list[float]], dense_limit: int, bm25_limit: int, limit: int,
                      doc_ids: Sequence[str] | None = None) -> list[RetrievedChunk]:
        """BM25 + dense in one round trip, fused with Reciprocal Rank Fusion inside Qdrant."""
        flt = self._filter(doc_ids)
        prefetch = [models.Prefetch(query=list(vector), using=DENSE, limit=dense_limit, filter=flt)]
        if sparse[0]:
            prefetch.append(models.Prefetch(query=models.SparseVector(indices=sparse[0], values=sparse[1]), using=SPARSE, limit=bm25_limit, filter=flt))
        res = self.client.query_points(self.collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True)
        return [self._to_chunk(p, "rrf") for p in res.points]

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        offset = None
        fields = ["doc_id", "file_name", "file_type", "doc_title", "ingested_at"]
        while True:
            points, offset = self.client.scroll(self.collection, limit=1000, offset=offset, with_payload=fields, with_vectors=False)
            for p in points:
                entry = docs.setdefault(p.payload["doc_id"], {k: p.payload.get(k) for k in fields} | {"chunks": 0})
                entry["chunks"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d["file_name"] or "")

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

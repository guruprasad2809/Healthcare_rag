"""Hybrid retrieval: BM25 + dense vectors -> RRF fusion -> cross-encoder rerank."""
from __future__ import annotations

from typing import Sequence

from app.ingestion.embeddings import BM25SparseEncoder
from app.retrieval.vector_store import QdrantStore, RetrievedChunk

MODES = ("hybrid", "dense", "bm25")


class HybridRetriever:
    def __init__(self, store: QdrantStore, embedder, sparse_encoder: BM25SparseEncoder, reranker=None, dense_top_k: int = 30,
                 bm25_top_k: int = 30, fusion_top_k: int = 30, final_top_k: int = 5, min_rerank_score: float = 0.0):
        self.store, self.embedder, self.sparse_encoder, self.reranker = store, embedder, sparse_encoder, reranker
        self.dense_top_k, self.bm25_top_k, self.fusion_top_k = dense_top_k, bm25_top_k, fusion_top_k
        self.final_top_k, self.min_rerank_score = final_top_k, min_rerank_score

    def retrieve(self, query: str, top_k: int | None = None, doc_ids: Sequence[str] | None = None, mode: str = "hybrid",
                 rerank: bool = True) -> list[RetrievedChunk]:
        """Return the `top_k` most relevant chunks for `query`.

        mode   -- "hybrid" (default), or "dense" / "bm25" to test one side alone
        rerank -- set False to see the raw fusion order (useful for ablations)
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        top_k = top_k or self.final_top_k
        if mode == "bm25":
            candidates = self.store.bm25_search(self.sparse_encoder.encode_query(query), self.fusion_top_k, doc_ids)
        elif mode == "dense":
            candidates = self.store.dense_search(self.embedder.embed_query(query), self.fusion_top_k, doc_ids)
        else:
            candidates = self.store.hybrid_search(self.embedder.embed_query(query), self.sparse_encoder.encode_query(query),
                                                  self.dense_top_k, self.bm25_top_k, self.fusion_top_k, doc_ids)
        if not rerank or self.reranker is None:
            return candidates[:top_k]
        ranked = self.reranker.rerank(query, candidates, top_k)
        return [c for c in ranked if c.score >= self.min_rerank_score]

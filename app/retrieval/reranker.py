"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3."""
from __future__ import annotations

from typing import Sequence

from app.ingestion.embeddings import resolve_device
from app.retrieval.vector_store import RetrievedChunk


class BGEReranker:
    """Scores each (question, chunk) pair jointly. Slower than vector search but much more
    precise, so it only runs on the short candidate list from hybrid search."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "", batch_size: int = 16, hf_token: str = ""):
        self.model_name, self.device, self.batch_size, self.hf_token = model_name, device, batch_size, hf_token
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=resolve_device(self.device), max_length=512, token=self.hf_token or None)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance in [0, 1] for each text (sigmoid of the model's logit)."""
        import math

        raw = [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)]
        if any(s < 0 or s > 1 for s in raw):  # library returned logits instead of probabilities
            raw = [1 / (1 + math.exp(-s)) for s in raw]
        return raw

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        for chunk, score in zip(chunks, self.score(query, [c.context_text for c in chunks])):
            chunk.scores["rerank"] = score
            chunk.score = score
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]

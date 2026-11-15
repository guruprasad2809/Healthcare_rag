"""Test fixtures: the full pipeline wired with tiny fakes (no model downloads, no API keys)."""
import re

import pytest
from qdrant_client import QdrantClient

from app.backend.rag import RAGService
from app.config import ROOT_DIR
from app.ingestion.chunking import SemanticTitleChunker
from app.ingestion.embeddings import BM25SparseEncoder, HashingEmbedder, tokenize
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import QdrantStore

DOCS = ROOT_DIR / "data" / "documents"


class OverlapReranker:
    """Stands in for bge-reranker: scores by share of question words found in the chunk."""

    def rerank(self, query, chunks, top_n):
        q = set(tokenize(query))
        for c in chunks:
            c.score = len(q & set(tokenize(c.context_text))) / (len(q) or 1)
            c.scores["rerank"] = c.score
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]


class EchoLLM:
    """Stands in for Gemini: answers with the first sentence of passage [1]."""

    def generate(self, prompt, system=""):
        m = re.search(r"\[1\] Source: .*?\nSection: .*?\n(.+?)(?:\n\n|$)", prompt, flags=re.S)
        return f"{m.group(1).split('. ')[0]}. [1]" if m else "The uploaded documents do not contain this information."


@pytest.fixture(scope="session")
def components():
    embedder, sparse = HashingEmbedder(256), BM25SparseEncoder()
    store = QdrantStore(QdrantClient(":memory:"), "test_docs", embedder.dim)
    pipeline = IngestionPipeline(embedder, sparse, store, SemanticTitleChunker())
    retriever = HybridRetriever(store, embedder, sparse, OverlapReranker(), min_rerank_score=0.3)
    return {"store": store, "pipeline": pipeline, "retriever": retriever, "rag": RAGService(retriever, EchoLLM())}


@pytest.fixture(scope="session")
def indexed(components):
    """All 15 sample documents ingested once."""
    for path in sorted(DOCS.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".txt", ".csv"}:
            components["pipeline"].ingest_file(path)
    return components

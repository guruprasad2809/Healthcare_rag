"""Builds each component once and shares it (models are heavy; load them a single time)."""
from functools import lru_cache

from app.backend.llm import GeminiLLM
from app.backend.rag import RAGService
from app.config import get_settings
from app.ingestion.chunking import SemanticTitleChunker
from app.ingestion.embeddings import BM25SparseEncoder, QwenEmbedder
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import BGEReranker
from app.retrieval.vector_store import QdrantStore, make_client


@lru_cache
def get_embedder() -> QwenEmbedder:
    s = get_settings()
    return QwenEmbedder(s.embedding_model, s.device, s.embedding_batch_size, s.hf_token)


@lru_cache
def get_sparse_encoder() -> BM25SparseEncoder:
    s = get_settings()
    return BM25SparseEncoder(s.bm25_k1, s.bm25_b, s.bm25_avg_len)


@lru_cache
def get_reranker() -> BGEReranker:
    s = get_settings()
    return BGEReranker(s.reranker_model, s.device, hf_token=s.hf_token)


@lru_cache
def get_store() -> QdrantStore:
    s = get_settings()
    return QdrantStore(make_client(s.qdrant_url, s.qdrant_api_key, s.qdrant_local_path), s.qdrant_collection, s.embedding_dim)


@lru_cache
def get_ingestion() -> IngestionPipeline:
    s = get_settings()
    embedder = get_embedder()
    chunker = SemanticTitleChunker(s.chunk_max_tokens, s.chunk_min_tokens, s.chunk_overlap_sentences,
                                   embed_fn=embedder.embed_documents if s.chunk_semantic_split else None)
    return IngestionPipeline(embedder, get_sparse_encoder(), get_store(), chunker)


@lru_cache
def get_retriever() -> HybridRetriever:
    s = get_settings()
    return HybridRetriever(get_store(), get_embedder(), get_sparse_encoder(), get_reranker(), s.dense_top_k, s.bm25_top_k,
                           s.fusion_top_k, s.final_top_k, s.min_rerank_score)


@lru_cache
def get_llm() -> GeminiLLM:
    s = get_settings()
    return GeminiLLM(s.gemini_api_key, s.gemini_model, s.llm_temperature, s.llm_max_output_tokens)


@lru_cache
def get_rag() -> RAGService:
    return RAGService(get_retriever(), get_llm())

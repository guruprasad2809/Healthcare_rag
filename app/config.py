"""Central configuration. Every value can be overridden in `.env` (see `.env.example`)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Gemini) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"  # switch to the Gemini 4 Hydrogen model ID once Google publishes it
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 2048

    # --- Embeddings / reranker (run locally, downloaded from Hugging Face) ---
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = ""  # "", "cpu", "cuda" or "mps"; empty = auto-detect
    hf_token: str = ""

    # --- Qdrant ---
    qdrant_url: str = ""  # empty = embedded local storage at qdrant_local_path
    qdrant_api_key: str = ""
    qdrant_local_path: str = str(ROOT_DIR / "qdrant_storage")
    qdrant_collection: str = "healthcare_docs"

    # --- Chunking (sizes are approximate tokens) ---
    chunk_max_tokens: int = 350
    chunk_min_tokens: int = 60
    chunk_overlap_sentences: int = 1
    chunk_semantic_split: bool = True  # use embeddings to find topic shifts inside long sections

    # --- Retrieval ---
    dense_top_k: int = 30
    bm25_top_k: int = 30
    fusion_top_k: int = 30  # candidates passed to the reranker
    final_top_k: int = 5  # chunks passed to the LLM
    min_rerank_score: float = 0.01  # drop chunks the reranker scores below this
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_len: float = 200.0

    # --- App ---
    upload_dir: str = str(ROOT_DIR / "data" / "uploads")
    max_upload_mb: int = 25
    backend_url: str = "http://localhost:8000"
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()

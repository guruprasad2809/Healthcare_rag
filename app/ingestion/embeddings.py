"""Dense embeddings (Qwen3-Embedding-0.6B) and sparse BM25 vectors."""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")  # keeps "covid-19", "b12", "2.5"
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his how i if in into is it its of on or s she that the "
    "their them there these they this to was were what when where which who whom why will with you your do does "
    "did can could should would about than then so such not no".split()
)


def resolve_device(device: str = "") -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenEmbedder:
    """Wraps `Qwen/Qwen3-Embedding-0.6B` (1024-dim). The model is loaded on first use.

    Qwen3 embeddings are instruction-aware: queries get the model's built-in
    "query" prompt, documents are embedded as-is.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "", batch_size: int = 8, hf_token: str = ""):
        self.model_name, self.device, self.batch_size, self.hf_token = model_name, device, batch_size, hf_token
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=resolve_device(self.device), token=self.hf_token or None)
            self._model.max_seq_length = 1024
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False)[0].tolist()


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class BM25SparseEncoder:
    """Turns text into sparse vectors so that Qdrant can score BM25 itself.

    BM25 = IDF x saturated term frequency. Qdrant applies the IDF part
    (`Modifier.IDF` on the sparse vector), so documents only need the
    term-frequency part and queries a 1.0 per term. No separate BM25 index has
    to be kept in memory or rebuilt when documents are added or deleted.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 200.0):
        self.k1, self.b, self.avg_len = k1, b, avg_len

    @staticmethod
    def _index(token: str) -> int:
        return zlib.crc32(token.encode("utf-8"))  # stable uint32 id for a term

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            weights[self._index(token)] = tf * (self.k1 + 1) / (tf + norm)
        return list(weights), list(weights.values())

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        ids = sorted({self._index(t) for t in tokenize(text)})
        return ids, [1.0] * len(ids)


class HashingEmbedder:
    """Tiny deterministic embedder for tests and offline demos (no model download).

    Not for real use: it only captures word overlap, not meaning.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

"""Semantic Title-Based Chunking.

1. Titles define sections. Each chunk belongs to exactly one section and carries
   its full heading trail ("Diabetes: Patient Guide > Diabetes > What causes Diabetes?").
2. A section that fits the size budget stays whole.
3. A section that is too long is split where the *meaning* shifts: between the
   two neighbouring sentences that are least similar (embedding cosine similarity
   when an embedder is supplied, word overlap otherwise). Paragraph breaks are
   preferred.
4. Very small neighbouring sections under the same parent heading are merged so
   that one-line sections do not become noisy, context-free chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from app.ingestion.parsers import Element, ParsedDocument
from app.ingestion.preprocess import count_tokens, split_sentences

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Chunk:
    text: str  # body shown to the LLM and the user
    title_path: list[str]  # heading trail, outermost first
    doc_id: str = ""
    chunk_index: int = 0
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def section_title(self) -> str:
        return self.title_path[-1] if self.title_path else ""

    @property
    def embed_text(self) -> str:
        """Text that gets embedded and BM25-indexed: heading trail + body."""
        trail = " > ".join(self.title_path)
        return f"{trail}\n\n{self.text}" if trail else self.text


@dataclass
class _Section:
    title_path: list[str]
    blocks: list[str]
    page: int | None


class SemanticTitleChunker:
    def __init__(self, max_tokens: int = 350, min_tokens: int = 60, overlap_sentences: int = 1, embed_fn: EmbedFn | None = None):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_sentences = overlap_sentences
        self.embed_fn = embed_fn

    # ------------------------------------------------------------------ public
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in self._sections(doc):
            body = "\n\n".join(section.blocks)
            if count_tokens(body) <= self.max_tokens:
                chunks.append(Chunk(body, section.title_path, page=section.page))
                continue
            parts = self._split_long_section(section.blocks)
            for n, part in enumerate(parts, start=1):
                chunks.append(Chunk(part, section.title_path, page=section.page, metadata={"part": n, "parts": len(parts)}))
        chunks = self._merge_small(chunks)
        for i, c in enumerate(chunks):
            c.doc_id, c.chunk_index = doc.doc_id, i
            c.metadata["token_count"] = count_tokens(c.text)
        return chunks

    # ---------------------------------------------------------------- sections
    @staticmethod
    def _sections(doc: ParsedDocument) -> list[_Section]:
        """Walk the elements keeping a stack of open headings."""
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        current: _Section | None = None
        for el in doc.elements:
            if el.kind == "title":
                while stack and stack[-1][0] >= el.level:
                    stack.pop()
                stack.append((el.level, el.text))
                current = None
                continue
            if current is None:
                path = [t for _, t in stack]
                if not path or path[0] != doc.title:
                    path = [doc.title] + path  # every chunk knows which document it is from
                current = _Section(path, [], el.page)
                sections.append(current)
            current.blocks.append(el.text)
        return sections

    # ------------------------------------------------------------ semantic split
    def _split_long_section(self, blocks: list[str]) -> list[str]:
        sentences, para_end = [], []  # para_end[i] is True when sentence i closes a paragraph
        for block in blocks:
            sents = split_sentences(block) or [block]
            sents = [piece for s in sents for piece in self._hard_wrap(s)]
            sentences += sents
            para_end += [False] * (len(sents) - 1) + [True]
        if len(sentences) == 1:
            return sentences
        sims = self._boundary_similarity(sentences)
        scores = [s - (0.15 if para_end[i] else 0.0) for i, s in enumerate(sims)]  # lower = better place to cut
        tokens = [count_tokens(s) for s in sentences]
        spans = self._bisect(0, len(sentences), scores, tokens)

        parts = []
        for start, end in spans:
            lead = sentences[max(0, start - self.overlap_sentences):start] if start else []
            if sum(map(count_tokens, lead)) > self.min_tokens:
                lead = []  # overlap is a short bridge, never a second copy of a long sentence
            parts.append(" ".join(lead + sentences[start:end]))
        return parts

    def _bisect(self, start: int, end: int, scores: list[float], tokens: list[int]) -> list[tuple[int, int]]:
        """Recursively cut [start, end) at its weakest link until every piece fits."""
        total = sum(tokens[start:end])
        if total <= self.max_tokens or end - start <= 1:
            return [(start, end)]
        floor = min(self.min_tokens, total // 3)
        best, best_score, left = None, float("inf"), 0
        for cut in range(start + 1, end):  # cut falls between sentence cut-1 and cut
            left += tokens[cut - 1]
            if left < floor or total - left < floor:
                continue
            if scores[cut - 1] < best_score:
                best, best_score = cut, scores[cut - 1]
        if best is None:
            best = (start + end) // 2
        return self._bisect(start, best, scores, tokens) + self._bisect(best, end, scores, tokens)

    def _boundary_similarity(self, sentences: list[str]) -> list[float]:
        """Similarity between sentence i and i+1 (length n-1)."""
        if self.embed_fn is not None:
            vecs = np.asarray(self.embed_fn(sentences), dtype=float)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            return [float(vecs[i] @ vecs[i + 1]) for i in range(len(sentences) - 1)]
        bags = [set(_WORD.findall(s.lower())) for s in sentences]
        return [len(a & b) / (len(a | b) or 1) for a, b in zip(bags, bags[1:])]

    def _hard_wrap(self, sentence: str) -> list[str]:
        """A single 'sentence' longer than the budget (e.g. a huge list) is wrapped by words."""
        if count_tokens(sentence) <= self.max_tokens:
            return [sentence]
        words, size = sentence.split(), max(self.max_tokens // 2, 1)
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    # ------------------------------------------------------------------ merging
    def _merge_small(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold tiny sibling sections (same parent heading) into one chunk."""
        merged: list[Chunk] = []
        for chunk in chunks:
            prev = merged[-1] if merged else None
            if (
                prev is not None
                and "part" not in prev.metadata
                and "part" not in chunk.metadata
                and self._parent(prev) == chunk.title_path[:-1]
                and len(chunk.title_path) > 1
                and min(count_tokens(prev.text), count_tokens(chunk.text)) < self.min_tokens
                and count_tokens(prev.text) + count_tokens(chunk.text) <= self.max_tokens
            ):
                if not prev.metadata.get("merged"):
                    prev.text = f"{prev.title_path[-1]}:\n{prev.text}"
                    prev.title_path = prev.title_path[:-1]
                    prev.metadata["merged"] = True
                prev.text += f"\n\n{chunk.title_path[-1]}:\n{chunk.text}"
            else:
                merged.append(chunk)
        return self._fold_intros(merged)

    def _fold_intros(self, chunks: list[Chunk]) -> list[Chunk]:
        """A few intro lines directly under a heading (e.g. "Source: ...") join its first sub-section."""
        out: list[Chunk] = []
        carry: Chunk | None = None
        for chunk in chunks:
            if carry is not None:
                depth = len(carry.title_path)
                if (
                    (len(chunk.title_path) > depth or chunk.metadata.get("merged"))
                    and chunk.title_path[:depth] == carry.title_path
                    and chunk.metadata.get("part", 1) == 1
                    and count_tokens(carry.text) + count_tokens(chunk.text) <= self.max_tokens * 1.15
                ):
                    chunk.text = f"{carry.text}\n\n{chunk.text}"
                    chunk.page = carry.page or chunk.page
                else:
                    out.append(carry)
                carry = None
            if count_tokens(chunk.text) < self.min_tokens and "part" not in chunk.metadata and not chunk.metadata.get("merged"):
                carry = chunk
            else:
                out.append(chunk)
        if carry is not None:
            out.append(carry)
        return out

    @staticmethod
    def _parent(chunk: Chunk) -> list[str]:
        return chunk.title_path if chunk.metadata.get("merged") else chunk.title_path[:-1]


# --------------------------------------------------------------------- baselines
# Used only by scripts/compare_chunking.py to show what title-based chunking buys.
def flatten(doc: ParsedDocument) -> str:
    return "\n\n".join(e.text for e in doc.elements)


def fixed_size_chunks(text: str, size: int = 350, overlap: int = 50) -> list[str]:
    """Fixed-size chunking: every `size` words, ignoring all structure."""
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def recursive_chunks(text: str, size: int = 350, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    """Recursive character-style chunking: split on the coarsest separator that fits, then pack."""
    if count_tokens(text) <= size:
        return [text] if text.strip() else []
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return fixed_size_chunks(text, size, 0)
    pieces, rest = text.split(sep), separators[separators.index(sep) + 1:]
    out, buf = [], ""
    for piece in pieces:
        candidate = f"{buf}{sep}{piece}" if buf else piece
        if count_tokens(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if count_tokens(piece) > size:
            out += recursive_chunks(piece, size, rest or (" ",))
            buf = ""
        else:
            buf = piece
    if buf:
        out.append(buf)
    return out

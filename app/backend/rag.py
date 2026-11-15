"""The RAG pipeline: retrieve -> build prompt -> generate -> return answer with sources."""
from __future__ import annotations

import re
import time
from typing import Sequence

from app.backend.prompts import SYSTEM_PROMPT, USER_TEMPLATE, build_context
from app.retrieval.hybrid import HybridRetriever


class RAGService:
    def __init__(self, retriever: HybridRetriever, llm):
        self.retriever, self.llm = retriever, llm

    def answer(self, question: str, top_k: int | None = None, doc_ids: Sequence[str] | None = None, mode: str = "hybrid",
               rerank: bool = True) -> dict:
        t0 = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k, doc_ids=doc_ids, mode=mode, rerank=rerank)
        t1 = time.perf_counter()
        prompt = USER_TEMPLATE.format(context=build_context(chunks), question=question.strip())
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        t2 = time.perf_counter()

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [
            {"index": i, "file_name": c.file_name, "section": " > ".join(c.title_path), "page": c.page, "text": c.text,
             "score": round(c.score, 4), "scores": {k: round(v, 4) for k, v in c.scores.items()}, "cited": i in cited}
            for i, c in enumerate(chunks, start=1)
        ]
        return {"question": question, "answer": answer, "sources": sources, "contexts": [c.context_text for c in chunks],
                "timings": {"retrieval_s": round(t1 - t0, 2), "generation_s": round(t2 - t1, 2)}}

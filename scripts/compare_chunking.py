"""Measure fixed-size vs recursive vs semantic title-based chunking on the sample documents.

No model downloads are needed: retrieval here is BM25 only, so the numbers isolate the
effect of *chunking*. Run:  python -m scripts.compare_chunking
"""
import json
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.config import ROOT_DIR
from app.ingestion.chunking import SemanticTitleChunker, fixed_size_chunks, flatten, recursive_chunks
from app.ingestion.embeddings import BM25SparseEncoder, tokenize
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.ingestion.preprocess import count_tokens

SIZE = 350
DOCS = ROOT_DIR / "data" / "documents"
EVAL = ROOT_DIR / "evaluation" / "eval_dataset.jsonl"


def build_chunks() -> dict[str, list[tuple[str, str]]]:
    """strategy -> [(file_name, text to index)]"""
    out = {"fixed_size": [], "recursive": [], "semantic_title": []}
    chunker = SemanticTitleChunker(max_tokens=SIZE)
    for path in sorted(p for p in DOCS.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS):
        doc = parse_document(path)
        flat = flatten(doc)
        out["fixed_size"] += [(doc.file_name, t) for t in fixed_size_chunks(flat, int(SIZE / 1.3), 35)]
        out["recursive"] += [(doc.file_name, t) for t in recursive_chunks(flat, SIZE)]
        out["semantic_title"] += [(doc.file_name, c.embed_text) for c in chunker.chunk(doc)]
    return out


def coverage(gold: set[str], text: str) -> float:
    return len(gold & set(tokenize(text))) / (len(gold) or 1)


def evaluate(chunks: list[tuple[str, str]], questions: list[dict]) -> dict:
    enc = BM25SparseEncoder()
    client = QdrantClient(":memory:")
    client.create_collection("c", vectors_config={}, sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    points = []
    for i, (_, text) in enumerate(chunks):
        idx, val = enc.encode_document(text)
        points.append(models.PointStruct(id=i, vector={"bm25": models.SparseVector(indices=idx, values=val)}, payload={}))
    client.upsert("c", points)

    hit1 = hit5 = mrr = best_cov = density = 0.0
    for q in questions:
        gold = set(tokenize(" ".join(q["reference_contexts"])))
        relevant = lambda i: chunks[i][0] == q["source_file"] and coverage(gold, chunks[i][1]) >= 0.5
        best_cov += max(coverage(gold, t) for f, t in chunks if f == q["source_file"])
        idx, val = enc.encode_query(q["question"])
        res = client.query_points("c", query=models.SparseVector(indices=idx, values=val), using="bm25", limit=5).points
        ranks = [r for r, p in enumerate(res, start=1) if relevant(p.id)]
        hit1 += bool(ranks and ranks[0] == 1)
        hit5 += bool(ranks)
        mrr += 1 / ranks[0] if ranks else 0
        if ranks:  # how much of the first relevant chunk is actually the answer (signal vs. noise)
            found = set(tokenize(chunks[res[ranks[0] - 1].id][1]))
            density += len(gold & found) / (len(found) or 1)
    n = len(questions)
    sizes = [count_tokens(t) for _, t in chunks]
    mid = sum(1 for _, t in chunks if t[:1].islower()) / len(chunks)
    return {"chunks": len(chunks), "avg_tokens": round(sum(sizes) / len(sizes)), "starts_mid_sentence_pct": round(100 * mid, 1),
            "best_single_chunk_answer_coverage_pct": round(100 * best_cov / n, 1), "bm25_hit@1_pct": round(100 * hit1 / n, 1),
            "bm25_hit@5_pct": round(100 * hit5 / n, 1), "bm25_mrr@5": round(mrr / n, 3), "answer_density_pct": round(100 * density / n, 1)}


def main() -> None:
    questions = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q["reference_contexts"]]
    results = {name: evaluate(chunks, questions) for name, chunks in build_chunks().items()}
    cols = list(next(iter(results.values())))
    print(f"{len(questions)} answerable questions, chunk budget ~{SIZE} tokens\n")
    print("| strategy | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for name, r in results.items():
        print(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    out = ROOT_DIR / "evaluation" / "results" / "chunking_comparison.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

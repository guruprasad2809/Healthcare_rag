"""Evaluate the RAG pipeline with RAGAS (5 metrics).

Prerequisites: documents are ingested (python -m scripts.ingest_folder) and GEMINI_API_KEY is set.

Usage:
    python -m evaluation.run_ragas                 # all questions
    python -m evaluation.run_ragas --limit 10      # quick check
    python -m evaluation.run_ragas --mode dense --no-rerank   # ablation runs
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from app.backend.container import get_embedder, get_rag
from app.config import ROOT_DIR, get_settings

warnings.filterwarnings("ignore", category=DeprecationWarning)
EVAL_FILE = ROOT_DIR / "evaluation" / "eval_dataset.jsonl"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]


def generate_answers(rows: list[dict], mode: str, rerank: bool, pause: float) -> list[dict]:
    """Run every question through the real pipeline and keep answer + retrieved contexts."""
    rag = get_rag()
    samples = []
    for i, row in enumerate(rows, start=1):
        result = rag.answer(row["question"], mode=mode, rerank=rerank)
        samples.append({"user_input": row["question"], "response": result["answer"],
                        "retrieved_contexts": result["contexts"] or ["(nothing retrieved)"], "reference": row["ground_truth"]})
        print(f"[{i}/{len(rows)}] {row['question'][:70]}")
        time.sleep(pause)
    return samples


def score(samples: list[dict]) -> pd.DataFrame:
    from langchain_core.embeddings import Embeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    settings = get_settings()
    embedder = get_embedder()

    class QwenForRagas(Embeddings):  # reuse the local Qwen model, no extra embedding API needed
        def embed_documents(self, texts):
            return embedder.embed_documents(texts)

        def embed_query(self, text):
            return embedder.embed_documents([text])[0]

    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.gemini_api_key, temperature=0))
    result = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall(), AnswerCorrectness()],
        llm=judge,
        embeddings=LangchainEmbeddingsWrapper(QwenForRagas()),
        run_config=RunConfig(max_workers=4, timeout=180, max_retries=5),
    )
    return result.to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    ap.add_argument("--tag", default="", help="label for the output files, e.g. 'hybrid_rerank'")
    args = ap.parse_args()

    rows = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: args.limit] if args.limit else rows
    samples = generate_answers(rows, args.mode, not args.no_rerank, args.pause)
    df = score(samples)
    df.insert(0, "id", [r["id"] for r in rows])
    df["category"] = [r["category"] for r in rows]
    df["source_file"] = [r["source_file"] for r in rows]

    metrics = [m for m in METRIC_NAMES if m in df.columns]
    tag = args.tag or f"{args.mode}{'' if not args.no_rerank else '_norerank'}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"ragas_{tag}.csv", index=False)
    summary = {"questions": len(df), "mode": args.mode, "rerank": not args.no_rerank, "llm": get_settings().gemini_model,
               "overall": df[metrics].mean().round(3).to_dict(),
               "by_category": df.groupby("category")[metrics].mean().round(3).to_dict(orient="index")}
    (RESULTS_DIR / f"ragas_{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nOverall scores (0-1, higher is better):")
    for name, value in summary["overall"].items():
        print(f"  {name:20} {value:.3f}")
    print("\nBy category:\n", df.groupby("category")[metrics].mean().round(3).to_string())
    print(f"\nSaved to {RESULTS_DIR.relative_to(ROOT_DIR)}/ragas_{tag}.csv")


if __name__ == "__main__":
    main()

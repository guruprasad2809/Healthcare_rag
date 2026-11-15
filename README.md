# Healthcare RAG Assistant

Upload healthcare documents (PDF, DOCX, TXT, MD, CSV), ask questions, and get answers that come only
from those documents, with the source passages shown.

| Part | Choice |
|---|---|
| LLM | Gemini (model set by `GEMINI_MODEL`; see the note below on Gemini 4 Hydrogen) |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` (1024-dim, runs locally) |
| Reranker | `BAAI/bge-reranker-v2-m3` (runs locally) |
| Chunking | Semantic Title-Based Chunking |
| Retrieval | Hybrid search: BM25 + vector embeddings, fused with RRF, then reranked |
| Vector database | Qdrant (dense + sparse vectors in one collection) |
| Evaluation | RAGAS: faithfulness, answer relevancy, context precision, context recall, answer correctness |
| Backend / frontend | FastAPI / Streamlit |

## How it works

```
                 ┌────────────── Frontend (Streamlit) ──────────────┐
                 │  upload files            ask a question          │
                 └───────┬───────────────────────┬──────────────────┘
                         ▼                       ▼
                 ┌────────────── Backend (FastAPI) ─────────────────┐
                 │ POST /api/documents/upload     POST /api/query   │
                 └───────┬───────────────────────┬──────────────────┘
      DATA INGESTION     ▼                       ▼      DATA RETRIEVAL
  parse (PDF/DOCX/TXT/MD/CSV)            embed question (Qwen3) + BM25 terms
  clean text, extract metadata                   │
  semantic title-based chunking                  ▼
  embed chunks (Qwen3) + BM25 vectors    Qdrant hybrid search (dense + BM25, RRF)
            │                                    │
            └────────►  Qdrant  ◄────────────────┘
                                                 ▼
                                   rerank (bge-reranker-v2-m3) → top 5
                                                 ▼
                                   Gemini answers from those passages, with [n] citations
```

## Project structure

```
app/
  config.py              all settings, read from .env
  ingestion/             MODULE 2 - Data ingestion
    parsers.py             PDF, DOCX, TXT, MD, CSV -> titles + text blocks + metadata
    preprocess.py          text cleaning, sentence splitting, token counting
    chunking.py            SemanticTitleChunker (+ fixed-size / recursive baselines)
    embeddings.py          Qwen3 embedder, BM25 sparse encoder
    pipeline.py            file -> chunks -> vectors -> Qdrant
  retrieval/             MODULE 3 - Data retrieval
    vector_store.py        Qdrant collection, hybrid / dense / BM25 queries
    reranker.py            bge-reranker-v2-m3
    hybrid.py              HybridRetriever (search -> fuse -> rerank)
  backend/               MODULE 4 - Backend
    llm.py                 Gemini client
    prompts.py             system prompt and context formatting
    rag.py                 RAGService (retrieve -> prompt -> generate -> sources)
    container.py           builds and shares the components
    schemas.py, main.py    FastAPI models and routes
frontend/
  streamlit_app.py       MODULE 1 - Frontend
evaluation/
  eval_dataset.jsonl     67 test questions with reference answers
  run_ragas.py           RAGAS evaluation (5 metrics)
scripts/
  ingest_folder.py       index a whole folder
  compare_chunking.py    fixed-size vs recursive vs title-based, measured
data/documents/          15 sample healthcare documents (4 PDF, 4 DOCX, 4 TXT, 3 CSV)
docs/                    implementation plan, chunking analysis
tests/                   33 tests (no model download or API key needed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add your GEMINI_API_KEY
```

The first run downloads the two models from Hugging Face (about 1.2 GB for Qwen3-Embedding-0.6B and
2.3 GB for bge-reranker-v2-m3). A GPU is faster but a CPU works.

Qdrant needs no setup: with `QDRANT_URL` empty it stores data in `./qdrant_storage`. To use a Qdrant
server instead, run `docker compose up -d` and set `QDRANT_URL=http://localhost:6333`.

## Run

```bash
python -m scripts.ingest_folder                 # index the 15 sample documents (optional)
uvicorn app.backend.main:app --port 8000        # backend, API docs at http://localhost:8000/docs
streamlit run frontend/streamlit_app.py         # frontend, in a second terminal
```

With embedded Qdrant storage only one process can open the database at a time, so stop the backend
before running `ingest_folder` or the evaluation, or use the Qdrant server.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | status and model names |
| POST | `/api/documents/upload` | upload one or more files and index them |
| GET | `/api/documents` | list indexed documents |
| DELETE | `/api/documents/{doc_id}` | remove a document |
| POST | `/api/query` | `{"question": "...", "top_k": 5, "mode": "hybrid", "rerank": true, "doc_ids": null}` |

## Evaluate with RAGAS

```bash
python -m evaluation.run_ragas --limit 10       # quick check
python -m evaluation.run_ragas                  # all 67 questions
python -m evaluation.run_ragas --mode dense --no-rerank --tag dense_only    # ablation
```

Scores are saved to `evaluation/results/ragas_<tag>.csv` (one row per question) and
`ragas_<tag>_summary.json` (averages, overall and by question category). Gemini is the judge model and
the local Qwen3 model supplies the embeddings RAGAS needs, so no other API key is required.

| Metric | Question it answers |
|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved passages? |
| Answer relevancy | Does the answer address the question that was asked? |
| Context precision | Are the retrieved passages relevant, with the best ones ranked first? |
| Context recall | Did retrieval find everything the reference answer needs? |
| Answer correctness | Does the answer agree with the reference answer? |

## Chunking comparison

`python -m scripts.compare_chunking` measures the three chunking strategies on the sample documents
(BM25 only, so it needs no models). Results and the reasoning are in
[docs/CHUNKING_ANALYSIS.md](docs/CHUNKING_ANALYSIS.md).

## Tests

```bash
pytest -q
```

The tests run the whole pipeline (parsing, chunking, Qdrant hybrid search, API) with small stand-ins
for the embedding model, the reranker and Gemini, so they need no downloads and no API key.

## Notes

- **Gemini 4 Hydrogen**: at the time of writing (November 2026) Google has not released this model or
  published a model ID for it, so the default is `gemini-3.8-flash`. To switch, change one line in
  `.env`: `GEMINI_MODEL=<model id>`.
- **Scanned PDFs** have no text layer and are rejected with a clear message; run OCR on them first.
- **Not medical advice.** The assistant only reports what the uploaded documents say.

More detail: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)

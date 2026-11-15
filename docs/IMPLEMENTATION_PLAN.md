# Implementation plan

The project has four modules. Each one lists its tasks, the steps to build it, the tools used, and what
you should see when it works. Everything described here is already implemented in this repository.

## Module 1 - Frontend (`frontend/streamlit_app.py`)

**Goal:** a simple page where a user uploads healthcare files and asks questions.

| Task | Steps | Tools |
|---|---|---|
| Upload documents | 1. Add a multi-file uploader limited to pdf, docx, txt, md, csv. 2. On "Add to knowledge base", send the files to `POST /api/documents/upload`. 3. Show a success line (chunk count) or the error for each file. | Streamlit `file_uploader`, `requests` |
| Manage documents | 1. Load `GET /api/documents` into the sidebar. 2. A checkbox per document limits the search to the ticked ones. 3. A remove button calls `DELETE /api/documents/{id}`. | Streamlit sidebar |
| Chat | 1. Keep the conversation in `st.session_state`. 2. Send each question to `POST /api/query`. 3. Show the answer, then the source passages (file, section, page, relevance score, cited or not) in an expander. | `st.chat_input`, `st.chat_message` |
| Search settings | Slider for the number of passages, selector for hybrid / dense / BM25, toggle for the reranker. | Streamlit widgets |
| Error handling | Show a plain message when the backend is not running or returns an error. | `requests` exceptions |

**Expected outcome:** a user can upload several files of different formats, see them listed, ask a
question, and read an answer with its sources, without touching any code.

## Module 2 - Data ingestion (`app/ingestion/`)

**Goal:** turn any supported file into clean, well-labelled chunks stored in Qdrant.

| Task | Steps | Tools |
|---|---|---|
| File upload | 1. Check the extension and size. 2. Save the file under `data/uploads/`. 3. Use a hash of the file content as `doc_id`, so re-uploading the same file replaces it instead of duplicating it. | FastAPI `UploadFile`, `hashlib` |
| Parsing and text extraction | One parser per format, all returning the same list of `Element`s (title or text). PDF: read lines with font size and bold flags; larger or bold short lines are titles. DOCX: use the Heading / Title styles; table rows become "column: value" text. TXT / MD: `#` headings, underlined headings, ALL-CAPS lines, short question lines, "Label:" lines. CSV: each row is a record; a name column (disease, drug, topic...) is its title and a question column its sub-title. | PyMuPDF, python-docx, pandas |
| Preprocessing | 1. Normalise unicode and ligatures. 2. Join words hyphenated across lines. 3. Collapse whitespace. 4. Drop running headers, footers and page numbers that repeat across PDF pages. | `unicodedata`, regex |
| Metadata extraction | Per document: id, file name, type, title, size, page or row count, ingestion time. Per chunk: heading trail, section title, page, position, token count, part number. | dataclasses |
| Semantic title-based chunking | 1. Group text under its heading trail. 2. Keep a section whole if it fits 350 tokens. 3. Split a longer section between the two least similar neighbouring sentences (Qwen3 embeddings), preferring paragraph breaks, until every piece fits. 4. Merge tiny neighbouring sections under the same parent. 5. Prefix each chunk with its heading trail before embedding. | `SemanticTitleChunker` |
| Embedding generation | 1. Dense vector from Qwen3-Embedding-0.6B (normalised, 1024-dim). 2. Sparse BM25 term-weight vector. 3. Upsert both, with the payload, into Qdrant in batches. | sentence-transformers, qdrant-client |

**Expected outcome:** `python -m scripts.ingest_folder` indexes the 15 sample documents into about
1,270 chunks, each one a single topic with its heading trail, source file and page.

## Module 3 - Data retrieval (`app/retrieval/`)

**Goal:** find the few passages that really answer the question.

| Task | Steps | Tools |
|---|---|---|
| Vector store | One Qdrant collection with two named vectors per chunk: `dense` (cosine) and `bm25` (sparse, IDF modifier). Keyword index on `doc_id` for filtering. | Qdrant |
| BM25 search | Tokenise, drop stop words, weight terms with the BM25 term-frequency formula at indexing time; Qdrant applies the IDF part at query time. Nothing has to be rebuilt when documents are added or removed. | `BM25SparseEncoder` |
| Dense search | Embed the question with the Qwen3 "query" prompt and search the `dense` vector. | `QwenEmbedder` |
| Hybrid fusion | One Qdrant query with two prefetches (top 30 dense, top 30 BM25) fused by Reciprocal Rank Fusion. | Qdrant Query API |
| Reranking | Score each (question, passage) pair with bge-reranker-v2-m3, keep the top 5, and drop passages below a minimum score so off-topic questions retrieve nothing. | sentence-transformers `CrossEncoder` |
| Options | `mode` = hybrid / dense / bm25 and `rerank` on or off, for ablation tests; `doc_ids` to search selected documents only. | `HybridRetriever.retrieve` |

**Expected outcome:** for a question such as "Who is at risk for glaucoma?" the top passage is the
matching section of the eye health guide, and for "Who won the cricket World Cup?" nothing is returned.

## Module 4 - Backend (`app/backend/`)

**Goal:** run the full RAG pipeline and connect the frontend to ingestion and retrieval.

| Task | Steps | Tools |
|---|---|---|
| LLM integration | Gemini client with the model name and key read from `.env`, low temperature, retries with back-off, and a clear error if the key is missing. | google-genai |
| Prompting | System prompt: answer only from the passages, cite them as [n], say so when the documents do not contain the answer, never diagnose or invent a dose, point emergencies to emergency services. Passages are numbered and labelled with file, page and section. | `prompts.py` |
| RAG pipeline | retrieve -> build prompt -> generate -> return the answer, the sources (with which ones were cited) and timings. | `RAGService` |
| API | Routes for health, upload, list, delete and query; request validation; errors mapped to HTTP codes. | FastAPI, pydantic |
| Component wiring | Each heavy object (models, Qdrant client) is built once and shared; tests swap in light stand-ins through FastAPI dependency overrides. | `container.py` |
| Evaluation | Run the 67 test questions through the pipeline and score them with the five RAGAS metrics. | RAGAS, langchain-google-genai |

**Expected outcome:** `POST /api/query` returns a grounded answer with numbered sources in a few
seconds, and `python -m evaluation.run_ragas` prints the five scores overall and by question category.

## How the modules fit together

1. The frontend only talks to the backend API. It holds no model code.
2. The backend calls two entry points: `IngestionPipeline.ingest_file()` and `RAGService.answer()`.
3. Ingestion and retrieval share the same embedder, sparse encoder and Qdrant store, so documents and
   questions are always encoded the same way.
4. All settings live in `app/config.py` and can be changed in `.env` without editing code.

## Scaling up later

- Run Qdrant as a server or in Qdrant Cloud (`QDRANT_URL`), and several backend workers behind it.
- Move ingestion to a background job queue when uploads get large.
- Add a parser for a new format by writing one function that returns `Element`s.
- Swap the LLM, embedder or reranker by changing one class in `container.py`.

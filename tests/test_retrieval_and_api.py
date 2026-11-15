from fastapi.testclient import TestClient

from app.backend import container
from app.backend.main import app
from tests.conftest import DOCS


def top_files(retriever, q, **kw):
    return [c.file_name for c in retriever.retrieve(q, **kw)]


def test_hybrid_search_finds_the_right_document(indexed):
    r = indexed["retriever"]
    assert top_files(r, "Who is at risk for Glaucoma?")[0] == "07_eye_health_guide.docx"
    assert top_files(r, "What precautions should be taken for Malaria?")[0] == "13_disease_descriptions_precautions.csv"
    hits = r.retrieve("Which operations were performed in the D&C discharge summary?")
    assert hits[0].file_name == "10_discharge_summaries.txt" and "Myomectomy" in " ".join(h.text for h in hits)


def test_modes_filters_and_scores(indexed):
    r = indexed["retriever"]
    for mode in ("hybrid", "dense", "bm25"):
        hits = r.retrieve("How to diagnose COPD?", mode=mode, rerank=False, top_k=5)
        assert len(hits) == 5 and "03_lung_health_guide.pdf" in [h.file_name for h in hits]
    doc_id = next(d["doc_id"] for d in indexed["store"].list_documents() if d["file_name"].startswith("01_"))
    assert set(top_files(r, "kidney disease", doc_ids=[doc_id])) == {"01_diabetes_patient_guide.pdf"}
    hits = r.retrieve("How to prevent Asthma?")
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True) and "rerank" in hits[0].scores


def test_out_of_scope_question_retrieves_nothing(indexed):
    assert indexed["retriever"].retrieve("Who won the cricket world cup final?") == []
    out = indexed["rag"].answer("Who won the cricket world cup final?")
    assert out["sources"] == [] and "do not contain" in out["answer"]


def test_rag_answer_has_cited_sources(indexed):
    out = indexed["rag"].answer("What is the outlook for Parkinson's Disease?")
    assert out["answer"].endswith("[1]")
    assert out["sources"][0]["cited"] and out["sources"][0]["file_name"] == "15_neurology_and_mental_health_qa.csv"
    assert "Parkinson's Disease" in out["sources"][0]["section"]


def test_reingesting_a_file_does_not_duplicate_chunks(indexed):
    before = indexed["store"].count()
    indexed["pipeline"].ingest_file(DOCS / "12_infectious_diseases_factsheet.txt")
    assert indexed["store"].count() == before
    assert len(indexed["store"].list_documents()) == 15


def test_api_upload_list_query_delete(indexed, tmp_path):
    app.dependency_overrides[container.get_ingestion] = lambda: indexed["pipeline"]
    app.dependency_overrides[container.get_store] = lambda: indexed["store"]
    app.dependency_overrides[container.get_rag] = lambda: indexed["rag"]
    client = TestClient(app)
    try:
        assert client.get("/api/health").json()["status"] == "ok"
        note = "VITAMIN D GUIDE\n===============\n\nWho needs vitamin D?\nOlder adults with little sun exposure need vitamin D supplements for bone strength.\n"
        files = [("files", ("vitamin_d.txt", note.encode(), "text/plain")), ("files", ("virus.exe", b"x", "application/octet-stream"))]
        up = client.post("/api/documents/upload", files=files).json()
        assert up[0]["status"] == "ingested" and up[0]["chunks"] == 1 and up[1]["status"] == "failed"
        assert len(client.get("/api/documents").json()) == 16

        res = client.post("/api/query", json={"question": "Who needs vitamin D supplements?", "doc_ids": [up[0]["doc_id"]]}).json()
        assert res["sources"][0]["file_name"] == "vitamin_d.txt" and "vitamin D" in res["answer"]
        assert client.post("/api/query", json={"question": "x"}).status_code == 422

        assert client.delete(f"/api/documents/{up[0]['doc_id']}").status_code == 200
        assert len(client.get("/api/documents").json()) == 15
    finally:
        app.dependency_overrides.clear()

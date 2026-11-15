import pytest

from app.ingestion.chunking import SemanticTitleChunker, fixed_size_chunks, recursive_chunks
from app.ingestion.embeddings import BM25SparseEncoder, tokenize
from app.ingestion.parsers import Element, ParsedDocument, parse_document
from app.ingestion.preprocess import clean_text, count_tokens, split_sentences
from tests.conftest import DOCS

FILES = sorted(p for p in DOCS.iterdir() if p.suffix in {".pdf", ".docx", ".txt", ".csv"})


def test_there_are_15_sample_documents_in_four_formats():
    assert len(FILES) == 15
    assert {p.suffix for p in FILES} == {".pdf", ".docx", ".txt", ".csv"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_document_parses_into_titles_and_text(path):
    doc = parse_document(path)
    assert doc.title and len(doc.doc_id) == 16
    assert sum(e.kind == "title" for e in doc.elements) >= 5
    assert sum(e.kind == "text" for e in doc.elements) >= 5


def test_pdf_headings_and_footer_removal():
    doc = parse_document(DOCS / "01_diabetes_patient_guide.pdf")
    titles = [e.text for e in doc.elements if e.kind == "title"]
    assert "What causes Diabetes?" in titles and "Diabetic Retinopathy" in titles
    assert not any("sample document - Page" in e.text for e in doc.elements)


def test_docx_heading_levels():
    doc = parse_document(DOCS / "07_eye_health_guide.docx")
    levels = {e.text: e.level for e in doc.elements if e.kind == "title"}
    assert levels["Glaucoma"] == 1 and levels["What is Glaucoma?"] == 2


def test_txt_heading_rules():
    doc = parse_document(DOCS / "10_discharge_summaries.txt")
    levels = {e.text: e.level for e in doc.elements if e.kind == "title"}
    assert levels["HOSPITAL COURSE"] == 3
    assert any(t.startswith("DISCHARGE SUMMARY 1:") and l == 2 for t, l in levels.items())
    assert not any(t.startswith("PMID") for t in levels)


def test_csv_rows_get_title_from_name_column():
    doc = parse_document(DOCS / "13_disease_descriptions_precautions.csv")
    assert doc.title == "Disease Descriptions Precautions"
    assert "Malaria" in [e.text for e in doc.elements if e.kind == "title"]


def test_unsupported_and_empty_files_are_rejected(tmp_path):
    bad = tmp_path / "scan.xyz"
    bad.write_text("x")
    with pytest.raises(ValueError):
        parse_document(bad)
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")
    with pytest.raises(ValueError):
        parse_document(empty)


def _doc(elements):
    return ParsedDocument("d1", "t.txt", "txt", "Guide", elements)


def test_chunks_carry_heading_trail_and_respect_sections():
    doc = _doc([Element("title", "Asthma", 1), Element("title", "Symptoms", 2), Element("text", "Wheezing and coughing. " * 20),
                Element("title", "Treatment", 2), Element("text", "Inhaled corticosteroids are used daily. " * 20)])
    chunks = SemanticTitleChunker(max_tokens=200, min_tokens=20).chunk(doc)
    assert [c.title_path for c in chunks] == [["Guide", "Asthma", "Symptoms"], ["Guide", "Asthma", "Treatment"]]
    assert chunks[0].embed_text.startswith("Guide > Asthma > Symptoms")
    assert "corticosteroids" not in chunks[0].text  # sections never bleed into each other


def test_long_section_splits_at_topic_shift_and_within_budget():
    diet = "Eat more vegetables and whole grains every day. " * 12
    drugs = "Metformin lowers blood glucose by reducing liver output. " * 12
    doc = _doc([Element("title", "Diabetes care", 1), Element("text", diet + drugs)])
    chunks = SemanticTitleChunker(max_tokens=150, min_tokens=30, overlap_sentences=0).chunk(doc)
    assert len(chunks) >= 2
    assert all(c.metadata["token_count"] <= 150 for c in chunks)
    assert all(c.title_path == ["Guide", "Diabetes care"] for c in chunks)
    mixed = [c for c in chunks if "vegetables" in c.text and "Metformin" in c.text]
    assert not mixed  # the cut lands exactly where the topic changes


def test_tiny_sibling_sections_are_merged():
    doc = _doc([Element("title", "Report", 1), Element("title", "Complications", 2), Element("text", "None."),
                Element("title", "Blood loss", 2), Element("text", "Less than 10 mL.")])
    chunks = SemanticTitleChunker().chunk(doc)
    assert len(chunks) == 1 and "Complications:" in chunks[0].text and "Blood loss:" in chunks[0].text


def test_sample_documents_chunk_within_budget():
    chunker = SemanticTitleChunker(max_tokens=350)
    for path in FILES[:6]:
        for c in chunker.chunk(parse_document(path)):
            assert 0 < c.metadata["token_count"] <= 350 * 1.2


def test_baseline_chunkers_and_helpers():
    text = "One two three. " * 300
    assert all(len(c.split()) <= 100 for c in fixed_size_chunks(text, 100, 10))
    assert all(count_tokens(c) <= 100 for c in recursive_chunks(text, 100))
    assert clean_text("treat-\nment  ofﬁce") == "treatment office"
    assert split_sentences("Take 2.5 mg daily. Call Dr. Smith if needed.")[0] == "Take 2.5 mg daily."


def test_bm25_encoder():
    enc = BM25SparseEncoder()
    assert tokenize("What is the COVID-19 dose of B12?") == ["covid-19", "dose", "b12"]
    idx, val = enc.encode_document("insulin insulin glucose")
    weights = dict(zip(idx, val))
    q_idx, _ = enc.encode_query("insulin")
    assert weights[q_idx[0]] == max(val)  # repeated term weighs more, but saturates
    assert max(val) < 2 * min(val)

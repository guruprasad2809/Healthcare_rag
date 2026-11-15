"""Document parsers.

Every parser turns a file into the same thing: an ordered list of `Element`s,
each either a *title* (with a heading level) or a block of *text*. The chunker
only ever sees elements, so adding a new file format means adding one parser.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.preprocess import clean_text

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


@dataclass
class Element:
    kind: str  # "title" or "text"
    text: str
    level: int = 0  # heading depth for titles (1 = top)
    page: int | None = None


@dataclass
class ParsedDocument:
    doc_id: str
    file_name: str
    file_type: str
    title: str
    elements: list[Element]
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- PDF
def _parse_pdf(path: Path) -> tuple[list[Element], dict]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    lines = []  # (page, text, font size, bold, block id)
    for page_no, page in enumerate(doc, start=1):
        for b_idx, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                size = max(spans, key=lambda s: len(s["text"]))["size"]
                bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                lines.append((page_no, text, round(size, 1), bold, (page_no, b_idx)))

    # Running headers/footers repeat on most pages once digits are masked.
    n_pages = len(doc)
    if n_pages >= 3:
        mask = lambda t: re.sub(r"\d+", "#", t)
        seen = Counter()
        for page_no in range(1, n_pages + 1):
            seen.update({mask(t) for p, t, *_ in lines if p == page_no})
        repeated = {t for t, c in seen.items() if c >= 0.6 * n_pages}
        lines = [l for l in lines if mask(l[1]) not in repeated]
    lines = [l for l in lines if not re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", l[1].lower())]

    # Body size = the size most characters are set in; larger (or bold, short) lines are titles.
    size_chars = Counter()
    for _, text, size, _, _ in lines:
        size_chars[size] += len(text)
    body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
    heading_sizes = sorted({s for s in size_chars if s >= body_size * 1.12}, reverse=True)

    def title_level(text: str, size: float, bold: bool) -> int:
        if len(text) > 160:
            return 0
        if size in heading_sizes:
            return min(heading_sizes.index(size) + 1, 4)
        if bold and len(text) <= 80 and not text.endswith((".", ",", ";")):
            return min(len(heading_sizes) + 1, 4)
        return 0

    elements: list[Element] = []
    para, para_block, para_page = [], None, None

    def flush():
        nonlocal para
        if para:
            elements.append(Element("text", clean_text(" ".join(para)), page=para_page))
            para = []

    for page_no, text, size, bold, block in lines:
        level = title_level(text, size, bold)
        if level:
            flush()
            last = elements[-1] if elements else None
            if last and last.kind == "title" and last.level == level and last.page == page_no and para_block == block:
                last.text = f"{last.text} {text}"  # heading wrapped over two lines
            else:
                elements.append(Element("title", clean_text(text), level=level, page=page_no))
            para_block = block
            continue
        if para and block != para_block and not _continues(para[-1], text):
            flush()
        if not para:
            para_page = page_no
        para.append(text)
        para_block = block
    flush()
    meta = {"n_pages": n_pages, "pdf_title": (doc.metadata or {}).get("title") or ""}
    doc.close()
    return elements, meta


def _continues(prev: str, nxt: str) -> bool:
    """True when a paragraph was split across a page/block boundary mid-sentence."""
    return not prev.rstrip().endswith((".", "!", "?", ":")) and nxt[:1].islower()


# -------------------------------------------------------------------------- DOCX
def _parse_docx(path: Path) -> tuple[list[Element], dict]:
    from docx import Document
    from docx.table import Table

    doc = Document(str(path))
    elements: list[Element] = []
    for item in doc.iter_inner_content():
        if isinstance(item, Table):
            rows = [[clean_text(c.text) for c in r.cells] for r in item.rows]
            if not rows:
                continue
            header = rows[0]
            for row in rows[1:]:
                pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
                if pairs:
                    elements.append(Element("text", "; ".join(pairs)))
            continue
        text = clean_text(item.text)
        if not text:
            continue
        style = (item.style.name or "").lower() if item.style is not None else ""
        if style == "title":
            elements.append(Element("title", text, level=1))
        elif style.startswith("heading"):
            digits = re.findall(r"\d+", style)
            elements.append(Element("title", text, level=min(int(digits[0]) if digits else 1, 6)))
        elif len(text) <= 80 and item.runs and all(r.bold for r in item.runs if r.text.strip()) and not text.endswith("."):
            elements.append(Element("title", text, level=4))  # bold one-liner used as a heading
        else:
            elements.append(Element("text", text))
    return elements, {"docx_title": doc.core_properties.title or ""}


# ---------------------------------------------------------------------- TXT / MD
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_UNDERLINE = re.compile(r"^(=+|-+)\s*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^.;,]{2,78})$")


def _is_caps_heading(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if ":" in line.rstrip(":"):
        return False  # "PMID: 123" is a key-value line, not a heading
    return 4 <= len(letters) and len(line) <= 100 and all(c.isupper() for c in letters)


def _parse_text(path: Path) -> tuple[list[Element], dict]:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in raw.split("\n")]
    elements: list[Element] = []
    para: list[str] = []

    def flush():
        nonlocal para
        if para:
            elements.append(Element("text", clean_text(" ".join(para))))
            para = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        prev_blank = i == 0 or not lines[i - 1].strip() or bool(_UNDERLINE.match(lines[i - 1].strip()))
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        level = 0
        title = line
        if not line:
            flush()
        elif _UNDERLINE.match(line):
            pass  # stray rule line
        elif m := _MD_HEADING.match(line):
            level, title = len(m.group(1)), m.group(2)
        elif nxt and _UNDERLINE.match(nxt) and len(nxt) >= 3 and len(line) <= 250 and not para:
            level = 1 if nxt.startswith("=") else 2  # setext / underlined heading
            i += 1
        elif _is_caps_heading(line) and not para:
            level = 3 if line.endswith(":") else 1  # "HOSPITAL COURSE:" vs "TUBERCULOSIS (TB)"
            title = line.rstrip(":")
        elif prev_blank and nxt and len(line) <= 120 and line.endswith("?"):
            level = 3  # a question on its own line, answered below
        elif prev_blank and len(line) <= 60 and line.endswith(":") and line[0].isupper() and nxt:
            level, title = 3, line.rstrip(":")  # "Symptoms:"
        elif prev_blank and (m := _NUMBERED.match(line)) and not _NUMBERED.match(nxt):
            level = min(m.group(1).count(".") + 1, 4)  # "2.1 Dosage"
        if level:
            flush()
            elements.append(Element("title", clean_text(title), level=level))
        elif line and not _UNDERLINE.match(line):
            para.append(line)
        i += 1
    flush()
    return elements, {}


# --------------------------------------------------------------------------- CSV
_TITLE_COLUMNS = ["disease", "condition", "drug", "medication", "topic", "focus", "title", "name", "subject", "sample_name"]
_SUBTITLE_COLUMNS = ["question", "query", "section", "section_title"]


def _parse_csv(path: Path) -> tuple[list[Element], dict]:
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    lower = {str(c).lower().strip(): c for c in df.columns}
    title_col = next((lower[c] for c in _TITLE_COLUMNS if c in lower), None)
    sub_col = next((lower[c] for c in _SUBTITLE_COLUMNS if c in lower and lower[c] != title_col), None)

    elements: list[Element] = []
    last_title = None
    for _, row in df.iterrows():
        title = clean_text(row[title_col]) if title_col else ""
        if title and title != last_title:
            elements.append(Element("title", title, level=1))
            last_title = title
        if sub_col and clean_text(row[sub_col]):
            elements.append(Element("title", clean_text(row[sub_col]), level=2))
        fields = [f"{c}: {clean_text(row[c])}" for c in df.columns if c not in (title_col, sub_col) and clean_text(row[c])]
        if fields:
            elements.append(Element("text", "\n".join(fields)))
    return elements, {"n_rows": len(df), "columns": [str(c) for c in df.columns], "title_column": title_col or ""}


_PARSERS = {".pdf": _parse_pdf, ".docx": _parse_docx, ".txt": _parse_text, ".md": _parse_text, ".csv": _parse_csv}


def parse_document(path: str | Path, file_name: str | None = None) -> ParsedDocument:
    """Parse any supported file into titles + text blocks, with document metadata."""
    path = Path(path)
    file_name = file_name or path.name
    ext = Path(file_name).suffix.lower()
    if ext not in _PARSERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    elements, extra = _PARSERS[ext](path)
    elements = [e for e in elements if e.text]
    if not any(e.kind == "text" for e in elements):
        raise ValueError(f"No readable text found in '{file_name}' (a scanned PDF needs OCR first).")

    data = path.read_bytes()
    fallback = re.sub(r"^\d+[_\-]", "", Path(file_name).stem).replace("_", " ").replace("-", " ").strip().title()
    first_title = next((e.text for e in elements[:3] if e.kind == "title"), "")
    if ext == ".csv":
        first_title = ""  # the first row's title is not the document's title
    title = extra.pop("pdf_title", "") or extra.pop("docx_title", "") or first_title or fallback
    metadata = {
        "file_size_bytes": len(data),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_titles": sum(e.kind == "title" for e in elements),
        **extra,
    }
    return ParsedDocument(
        doc_id=hashlib.sha1(data).hexdigest()[:16],
        file_name=file_name,
        file_type=ext.lstrip("."),
        title=title,
        elements=elements,
        metadata=metadata,
    )

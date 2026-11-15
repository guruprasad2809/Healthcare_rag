"""Text cleaning shared by all parsers."""
import re
import unicodedata

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "­": ""}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")
_TOKEN = re.compile(r"\w+|[^\w\s]")


def clean_text(text: str) -> str:
    """Normalise unicode, undo PDF hyphenation and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # hyphen-ated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def count_tokens(text: str) -> int:
    """Cheap, model-free token estimate (words + punctuation marks)."""
    return len(_TOKEN.findall(text))

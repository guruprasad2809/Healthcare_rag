"""Request/response models for the API."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None  # restrict the search to these documents
    mode: str = Field(default="hybrid", pattern="^(hybrid|dense|bm25)$")
    rerank: bool = True


class Source(BaseModel):
    index: int
    file_name: str
    section: str
    page: int | None = None
    text: str
    score: float
    scores: dict[str, float] = {}
    cited: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    timings: dict[str, float] = {}


class DocumentInfo(BaseModel):
    doc_id: str
    file_name: str
    file_type: str | None = None
    doc_title: str | None = None
    ingested_at: str | None = None
    chunks: int = 0


class UploadResult(BaseModel):
    file_name: str
    status: str  # "ingested" or "failed"
    doc_id: str | None = None
    chunks: int = 0
    seconds: float = 0.0
    error: str | None = None

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, with its source reference and score."""

    text: str
    source: str
    document_id: str
    chunk_index: int
    score: float


class RetrievalResult(BaseModel):
    """Outcome of a retrieval. When has_context is False, chunks is empty."""

    has_context: bool
    chunks: list[RetrievedChunk]

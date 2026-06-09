from pydantic import BaseModel, Field

from app.retrieval.schemas import RetrievedChunk


class ChatRequest(BaseModel):
    """A user's question."""

    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """The grounded answer with its sources.

    `grounded` is True when the answer came from retrieved context, and False
    when no relevant context was found (the safe fallback).
    """

    answer: str
    grounded: bool
    sources: list[RetrievedChunk]

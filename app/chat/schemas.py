from pydantic import BaseModel, Field

from app.agent.schemas import AgentStep
from app.retrieval.schemas import RetrievedChunk


class ChatRequest(BaseModel):
    """A user's question."""

    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """The agent's answer with its sources and decision trace.

    `grounded` is True when the answer came from retrieved context. `escalated` is
    True when no context was found even after a re-query (handed off to a human).
    """

    answer: str
    grounded: bool
    escalated: bool
    sources: list[RetrievedChunk]
    decisions: list[AgentStep]

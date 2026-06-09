from enum import StrEnum

from pydantic import BaseModel


class AgentAction(StrEnum):
    """The three — and only three — actions the agent may take."""

    ANSWER = "answer"
    REQUERY = "requery"
    ESCALATE = "escalate"


class AgentStep(BaseModel):
    """One decision the agent made, with the reason it made it (inspectable)."""

    action: AgentAction
    reason: str

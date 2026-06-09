import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.reformulate import reformulate_query
from app.agent.schemas import AgentAction, AgentStep
from app.chat.prompts import FALLBACK_ANSWER
from app.chat.schemas import ChatResponse
from app.chat.service import generate_answer, log_qa
from app.clients.claude import ClaudeClient
from app.clients.qdrant import QdrantClientT
from app.embeddings.client import EmbeddingClient
from app.retrieval.service import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# Escalation keeps the honest fallback wording and adds a clear human-handoff signal.
ESCALATION_ANSWER = f"{FALLBACK_ANSWER} It has been flagged for a human to follow up."


async def run_agent(
    session: AsyncSession,
    qdrant: QdrantClientT,
    embedder: EmbeddingClient,
    claude: ClaudeClient,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int,
    score_threshold: float,
    session_id: uuid.UUID | None = None,
) -> ChatResponse:
    """Retrieve, then take exactly one of three actions: answer, re-query, or escalate.

    Re-query happens at most once (a deterministic keyword reformulation, no LLM),
    so there are no loops. Every decision is logged with its reason.
    """
    decisions: list[AgentStep] = []

    result = await retrieve_relevant_chunks(
        qdrant, embedder, tenant_id, question, top_k, score_threshold
    )

    if result.has_context:
        decisions.append(
            AgentStep(
                action=AgentAction.ANSWER, reason="relevant context found on the first retrieval"
            )
        )
    else:
        decisions.append(
            AgentStep(
                action=AgentAction.REQUERY,
                reason="no relevant context; reformulating the query once",
            )
        )
        reformulated = reformulate_query(question)
        if reformulated != question:
            result = await retrieve_relevant_chunks(
                qdrant, embedder, tenant_id, reformulated, top_k, score_threshold
            )
        if result.has_context:
            decisions.append(
                AgentStep(
                    action=AgentAction.ANSWER, reason="relevant context found after one re-query"
                )
            )
        else:
            decisions.append(
                AgentStep(
                    action=AgentAction.ESCALATE, reason="no relevant context after one re-query"
                )
            )

    for step in decisions:
        logger.info("agent decision: %s — %s", step.action.value, step.reason)

    if result.has_context:
        answer = await generate_answer(claude, question, result.chunks)
        await log_qa(
            session,
            tenant_id,
            question,
            answer,
            result.chunks,
            grounded=True,
            session_id=session_id,
        )
        return ChatResponse(
            answer=answer,
            grounded=True,
            escalated=False,
            sources=result.chunks,
            decisions=decisions,
        )

    await log_qa(
        session, tenant_id, question, ESCALATION_ANSWER, [], grounded=False, session_id=session_id
    )
    return ChatResponse(
        answer=ESCALATION_ANSWER,
        grounded=False,
        escalated=True,
        sources=[],
        decisions=decisions,
    )

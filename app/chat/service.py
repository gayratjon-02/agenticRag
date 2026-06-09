import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatLog
from app.chat.prompts import FALLBACK_ANSWER, GROUNDING_SYSTEM_PROMPT, build_user_prompt
from app.chat.schemas import ChatResponse
from app.clients.claude import ClaudeClient
from app.clients.qdrant import QdrantClientT
from app.embeddings.client import EmbeddingClient
from app.retrieval.schemas import RetrievedChunk
from app.retrieval.service import retrieve_relevant_chunks

logger = logging.getLogger(__name__)


async def answer_question(
    session: AsyncSession,
    qdrant: QdrantClientT,
    embedder: EmbeddingClient,
    claude: ClaudeClient,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int,
    score_threshold: float,
) -> ChatResponse:
    """Retrieve context, generate a grounded answer (or safe fallback), and log it.

    Claude is NEVER called without retrieved context attached (§4.4).
    """
    result = await retrieve_relevant_chunks(
        qdrant, embedder, tenant_id, question, top_k, score_threshold
    )

    if not result.has_context:
        await _log(session, tenant_id, question, FALLBACK_ANSWER, [], grounded=False)
        return ChatResponse(answer=FALLBACK_ANSWER, grounded=False, sources=[])

    user_prompt = build_user_prompt(question, result.chunks)
    answer = await claude.generate(GROUNDING_SYSTEM_PROMPT, user_prompt)
    await _log(session, tenant_id, question, answer, result.chunks, grounded=True)
    return ChatResponse(answer=answer, grounded=True, sources=result.chunks)


async def _log(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    grounded: bool,
) -> None:
    chunk_ids = [f"{chunk.document_id}:{chunk.chunk_index}" for chunk in chunks]
    session.add(
        ChatLog(
            tenant_id=tenant_id,
            question=question,
            answer=answer,
            chunk_ids=chunk_ids,
            grounded=grounded,
        )
    )
    await session.commit()

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.models import ChatLog
from app.chat.prompts import GROUNDING_SYSTEM_PROMPT, build_user_prompt
from app.clients.claude import ClaudeClient
from app.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


async def generate_answer(claude: ClaudeClient, question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer from retrieved chunks (context is always attached)."""
    return await claude.generate(GROUNDING_SYSTEM_PROMPT, build_user_prompt(question, chunks))


async def log_qa(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    grounded: bool,
    session_id: uuid.UUID | None = None,
) -> None:
    """Record a question/answer turn for the tenant (optionally within a session)."""
    chunk_ids = [f"{chunk.document_id}:{chunk.chunk_index}" for chunk in chunks]
    session.add(
        ChatLog(
            tenant_id=tenant_id,
            session_id=session_id,
            question=question,
            answer=answer,
            chunk_ids=chunk_ids,
            grounded=grounded,
        )
    )
    await session.commit()

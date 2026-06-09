from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.schemas import ChatRequest, ChatResponse
from app.chat.service import answer_question
from app.clients.claude import ClaudeClient, get_claude_client
from app.clients.qdrant import QdrantClientT, get_qdrant
from app.config import Settings, get_settings
from app.db.session import get_db
from app.embeddings.client import EmbeddingClient, get_embedding_client
from app.tenants.dependencies import CurrentTenant

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    tenant: CurrentTenant,
    session: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[QdrantClientT, Depends(get_qdrant)],
    embedder: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    claude: Annotated[ClaudeClient, Depends(get_claude_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    return await answer_question(
        session=session,
        qdrant=qdrant,
        embedder=embedder,
        claude=claude,
        tenant_id=tenant.id,
        question=payload.question,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )

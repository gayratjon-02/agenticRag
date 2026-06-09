import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.documents.models import Document, IngestionStatus


async def create_document(
    session: AsyncSession, tenant_id: uuid.UUID, source: str, content: str
) -> Document:
    """Persist a new document in the 'queued' state and return it."""
    document = Document(
        tenant_id=tenant_id,
        source=source,
        content=content,
        status=IngestionStatus.QUEUED,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def get_document(
    session: AsyncSession, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> Document | None:
    """Fetch a document scoped to its tenant. Cross-tenant access returns None."""
    result = await session.execute(
        select(Document).where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
        )
    )
    return result.scalars().one_or_none()


async def get_document_by_id(session: AsyncSession, document_id: uuid.UUID) -> Document | None:
    """Fetch a document by id only. For internal ingestion use, not request handlers."""
    result = await session.execute(select(Document).where(Document.id == document_id))
    return result.scalars().one_or_none()


async def set_status(
    session: AsyncSession,
    document: Document,
    status: IngestionStatus,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Update a document's ingestion status. Caller commits."""
    document.status = status
    document.error = error
    if chunk_count is not None:
        document.chunk_count = chunk_count

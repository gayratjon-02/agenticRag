import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.qdrant import (
    ChunkPoint,
    QdrantClientT,
    collection_name_for_tenant,
    ensure_collection,
    upsert_chunks,
)
from app.documents.chunker import chunk_text
from app.documents.models import IngestionStatus
from app.documents.service import get_document_by_id, set_status
from app.embeddings.client import EmbeddingClient

logger = logging.getLogger(__name__)

# Fixed namespace so a chunk's point id is stable across re-ingestion (upsert, no dupes).
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


async def run_ingestion(
    document_id: uuid.UUID,
    sessionmaker: async_sessionmaker[AsyncSession],
    qdrant: QdrantClientT,
    embedder: EmbeddingClient,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Chunk a document, embed it, and store the vectors in the tenant's collection.

    Marks the document processing → done, or failed (with the error) on any error.
    """
    async with sessionmaker() as session:
        document = await get_document_by_id(session, document_id)
        if document is None:
            logger.warning("Ingestion skipped: document %s not found", document_id)
            return
        tenant_id = document.tenant_id
        source = document.source
        content = document.content
        await set_status(session, document, IngestionStatus.PROCESSING)
        await session.commit()

    try:
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        vectors = await embedder.embed_batch(chunks)
        collection = collection_name_for_tenant(tenant_id)
        await ensure_collection(qdrant, collection, embedder.dimension)
        points = [
            ChunkPoint(
                id=str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{index}")),
                vector=vector,
                payload={
                    "tenant_id": str(tenant_id),
                    "document_id": str(document_id),
                    "source": source,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await upsert_chunks(qdrant, collection, points)
        await _finalize(sessionmaker, document_id, IngestionStatus.DONE, chunk_count=len(points))
    except Exception as exc:
        logger.exception("Ingestion failed for document %s", document_id)
        await _finalize(sessionmaker, document_id, IngestionStatus.FAILED, error=str(exc))
        raise


async def _finalize(
    sessionmaker: async_sessionmaker[AsyncSession],
    document_id: uuid.UUID,
    status: IngestionStatus,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
) -> None:
    async with sessionmaker() as session:
        document = await get_document_by_id(session, document_id)
        if document is None:
            return
        await set_status(session, document, status, error=error, chunk_count=chunk_count)
        await session.commit()

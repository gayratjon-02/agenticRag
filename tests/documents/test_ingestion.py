import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.qdrant import (
    collection_name_for_tenant,
    create_qdrant_client,
    delete_collection,
)
from app.config import get_settings
from app.documents.ingestion import run_ingestion
from app.documents.models import Document, IngestionStatus
from app.documents.service import get_document_by_id
from app.embeddings.client import EmbeddingClient, create_embedding_client
from app.tenants.models import Tenant
from app.tenants.security import generate_api_key, hash_api_key

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> EmbeddingClient:
    return create_embedding_client(get_settings())


async def _seed_document(
    sessionmaker: async_sessionmaker[AsyncSession], content: str
) -> tuple[uuid.UUID, uuid.UUID]:
    async with sessionmaker() as session:
        tenant = Tenant(name="Acme", api_key_hash=hash_api_key(generate_api_key()))
        session.add(tenant)
        await session.commit()
        await session.refresh(tenant)

        document = Document(
            tenant_id=tenant.id,
            source="doc.txt",
            content=content,
            status=IngestionStatus.QUEUED,
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        return tenant.id, document.id


async def test_ingestion_marks_done_and_stores_tenant_scoped_vectors(
    db_sessionmaker: async_sessionmaker[AsyncSession], embedder: EmbeddingClient
) -> None:
    settings = get_settings()
    qdrant = create_qdrant_client(settings)
    content = "Agentic RAG keeps each tenant's data isolated. " * 50
    tenant_id, document_id = await _seed_document(db_sessionmaker, content)
    collection = collection_name_for_tenant(tenant_id)
    try:
        await run_ingestion(
            document_id=document_id,
            sessionmaker=db_sessionmaker,
            qdrant=qdrant,
            embedder=embedder,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        async with db_sessionmaker() as session:
            doc = await get_document_by_id(session, document_id)
        assert doc is not None
        assert doc.status == IngestionStatus.DONE
        assert doc.chunk_count > 0

        points, _ = await qdrant.scroll(collection_name=collection, limit=1000, with_payload=True)
        assert len(points) == doc.chunk_count
        for point in points:
            assert point.payload is not None
            assert point.payload["tenant_id"] == str(tenant_id)
            assert point.payload["document_id"] == str(document_id)
    finally:
        await delete_collection(qdrant, collection)
        await qdrant.close()


async def test_ingestion_marks_failed_and_records_error(
    db_sessionmaker: async_sessionmaker[AsyncSession],
    embedder: EmbeddingClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    qdrant = create_qdrant_client(settings)
    _, document_id = await _seed_document(db_sessionmaker, "some content to ingest")

    async def _boom(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(embedder, "embed_batch", _boom)
    try:
        with pytest.raises(RuntimeError):
            await run_ingestion(
                document_id=document_id,
                sessionmaker=db_sessionmaker,
                qdrant=qdrant,
                embedder=embedder,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

        async with db_sessionmaker() as session:
            doc = await get_document_by_id(session, document_id)
        assert doc is not None
        assert doc.status == IngestionStatus.FAILED
        assert doc.error  # error recorded, not swallowed
    finally:
        await qdrant.close()

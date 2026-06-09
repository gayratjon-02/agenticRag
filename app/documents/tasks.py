import asyncio
import logging
import uuid
from functools import lru_cache

from app.celery_app import celery_app
from app.clients.qdrant import create_qdrant_client
from app.config import get_settings
from app.db.session import create_engine, create_sessionmaker
from app.documents.ingestion import run_ingestion
from app.embeddings.client import EmbeddingClient, create_embedding_client

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _worker_embedder() -> EmbeddingClient:
    """Load the embedding model once per worker process and reuse it across tasks."""
    return create_embedding_client(get_settings())


# reason: celery's task decorator is untyped; the function below is fully typed.
@celery_app.task(name="documents.ingest")  # type: ignore[untyped-decorator]
def ingest_document_task(document_id: str) -> None:
    """Celery entrypoint: run the async ingestion pipeline for one document."""
    asyncio.run(_run(uuid.UUID(document_id)))


async def _run(document_id: uuid.UUID) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    qdrant = create_qdrant_client(settings)
    try:
        await run_ingestion(
            document_id=document_id,
            sessionmaker=sessionmaker,
            qdrant=qdrant,
            embedder=_worker_embedder(),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    finally:
        await qdrant.close()
        await engine.dispose()

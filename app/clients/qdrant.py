import logging
import uuid
from dataclasses import dataclass

from fastapi import Request
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.common import Status
from app.config import Settings

logger = logging.getLogger(__name__)

# Re-exported so other modules can type the client without importing the SDK directly.
QdrantClientT = AsyncQdrantClient


@dataclass
class ChunkPoint:
    """A single vector to store: its id, embedding, and payload metadata."""

    id: str
    vector: list[float]
    payload: dict[str, object]


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create the async Qdrant client. Only place the qdrant SDK is constructed."""
    # Treat an empty key as "no key" so local http connections don't warn.
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def collection_name_for_tenant(tenant_id: uuid.UUID) -> str:
    """One collection per tenant — the hard boundary for multi-tenant isolation."""
    return f"tenant_{tenant_id}"


async def ensure_collection(client: AsyncQdrantClient, name: str, dimension: int) -> None:
    """Create the tenant's collection if it does not already exist."""
    if await client.collection_exists(name):
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )


async def upsert_chunks(client: AsyncQdrantClient, name: str, points: list[ChunkPoint]) -> None:
    """Store chunk vectors in a tenant's collection."""
    if not points:
        return
    await client.upsert(
        collection_name=name,
        points=[
            PointStruct(id=point.id, vector=point.vector, payload=point.payload) for point in points
        ],
    )


async def delete_collection(client: AsyncQdrantClient, name: str) -> None:
    """Delete a tenant's collection (used for cleanup/tests)."""
    await client.delete_collection(collection_name=name)


async def qdrant_health(request: Request) -> Status:
    """Probe Qdrant connectivity. Returns DOWN instead of raising on failure."""
    client: AsyncQdrantClient = request.app.state.qdrant
    try:
        await client.get_collections()
        return Status.OK
    except Exception:
        logger.exception("Qdrant health check failed")
        return Status.DOWN

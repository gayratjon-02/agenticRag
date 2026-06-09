import logging

from fastapi import Request
from qdrant_client import AsyncQdrantClient

from app.common import Status
from app.config import Settings

logger = logging.getLogger(__name__)


def create_qdrant_client(settings: Settings) -> AsyncQdrantClient:
    """Create the async Qdrant client. Only place the qdrant SDK is constructed."""
    # Treat an empty key as "no key" so local http connections don't warn.
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


async def qdrant_health(request: Request) -> Status:
    """Probe Qdrant connectivity. Returns DOWN instead of raising on failure."""
    client: AsyncQdrantClient = request.app.state.qdrant
    try:
        await client.get_collections()
        return Status.OK
    except Exception:
        logger.exception("Qdrant health check failed")
        return Status.DOWN

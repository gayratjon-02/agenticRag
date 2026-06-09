import asyncio
import logging

from fastapi import Request
from fastembed import TextEmbedding

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingDimensionError(RuntimeError):
    """Raised when a produced embedding does not match the configured dimension."""


class EmbeddingClient:
    """The single wrapper around the embedding model.

    This is the ONLY place the fastembed SDK is used. The same model embeds both
    documents (ingestion) and questions (query) — mixing models would corrupt
    retrieval, so the model name and dimension come from settings and every vector
    is checked against the configured dimension.
    """

    def __init__(self, model_name: str, dimension: int) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model = TextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        vectors = [vector.tolist() for vector in self._model.embed(texts)]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingDimensionError(
                    f"Model '{self._model_name}' returned dimension {len(vector)}, "
                    f"expected {self._dimension}"
                )
        return vectors

    async def embed(self, text: str) -> list[float]:
        """Embed a single piece of text into a vector of the configured dimension."""
        vectors = await asyncio.to_thread(self._embed_sync, [text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts at once. Returns one vector per input, in order."""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    return EmbeddingClient(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )


def get_embedding_client(request: Request) -> EmbeddingClient:
    """FastAPI dependency that returns the shared embedding client."""
    client: EmbeddingClient = request.app.state.embedding
    return client

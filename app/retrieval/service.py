import logging
import uuid
from typing import cast

from app.clients.qdrant import (
    QdrantClientT,
    SearchHit,
    collection_name_for_tenant,
    get_vector_dimension,
    search,
)
from app.embeddings.client import EmbeddingClient, EmbeddingDimensionError
from app.retrieval.schemas import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    qdrant: QdrantClientT,
    embedder: EmbeddingClient,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int,
    score_threshold: float,
) -> RetrievalResult:
    """Embed the question and return the tenant's most relevant chunks.

    Always isolated to one tenant. If the tenant has no indexed documents, or no
    chunk scores at or above `score_threshold`, returns has_context=False.
    """
    if not question.strip():
        return RetrievalResult(has_context=False, chunks=[])

    collection = collection_name_for_tenant(tenant_id)
    dimension = await get_vector_dimension(qdrant, collection)
    if dimension is None:
        # No collection yet → the tenant has nothing indexed.
        return RetrievalResult(has_context=False, chunks=[])
    if dimension != embedder.dimension:
        # Stored vectors were built with a different model — refuse to mix (§4.2).
        raise EmbeddingDimensionError(
            f"Collection '{collection}' has dimension {dimension}, "
            f"but the embedding model produces {embedder.dimension}; re-index required"
        )

    query_vector = await embedder.embed(question)
    hits = await search(qdrant, collection, query_vector, top_k, tenant_id)
    relevant = [hit for hit in hits if hit.score >= score_threshold]
    if not relevant:
        return RetrievalResult(has_context=False, chunks=[])

    return RetrievalResult(has_context=True, chunks=[_to_chunk(hit) for hit in relevant])


def _to_chunk(hit: SearchHit) -> RetrievedChunk:
    payload = hit.payload
    return RetrievedChunk(
        text=str(payload.get("text", "")),
        source=str(payload.get("source", "")),
        document_id=str(payload.get("document_id", "")),
        chunk_index=cast(int, payload.get("chunk_index", 0)),
        score=hit.score,
    )

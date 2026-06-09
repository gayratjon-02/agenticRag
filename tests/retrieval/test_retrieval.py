import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.clients.qdrant import (
    ChunkPoint,
    QdrantClientT,
    collection_name_for_tenant,
    create_qdrant_client,
    delete_collection,
    ensure_collection,
    upsert_chunks,
)
from app.config import get_settings
from app.embeddings.client import (
    EmbeddingClient,
    EmbeddingDimensionError,
    create_embedding_client,
)
from app.retrieval.service import retrieve_relevant_chunks

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> EmbeddingClient:
    return create_embedding_client(get_settings())


@pytest_asyncio.fixture
async def qdrant() -> AsyncIterator[QdrantClientT]:
    client = create_qdrant_client(get_settings())
    yield client
    await client.close()


async def _seed(
    client: QdrantClientT, embedder: EmbeddingClient, tenant_id: uuid.UUID, texts: list[str]
) -> None:
    name = collection_name_for_tenant(tenant_id)
    await ensure_collection(client, name, embedder.dimension)
    vectors = await embedder.embed_batch(texts)
    points = [
        ChunkPoint(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "tenant_id": str(tenant_id),
                "document_id": str(uuid.uuid4()),
                "source": "doc.txt",
                "chunk_index": index,
                "text": text,
            },
        )
        for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]
    await upsert_chunks(client, name, points)


async def test_returns_relevant_chunks_with_sources_and_scores(
    qdrant: QdrantClientT, embedder: EmbeddingClient
) -> None:
    tenant = uuid.uuid4()
    collection = collection_name_for_tenant(tenant)
    try:
        await _seed(
            qdrant,
            embedder,
            tenant,
            [
                "The capital of France is Paris.",
                "Cats are popular pets that enjoy sleeping.",
                "Photosynthesis lets plants convert sunlight into energy.",
            ],
        )

        result = await retrieve_relevant_chunks(
            qdrant, embedder, tenant, "What is the capital of France?", top_k=3, score_threshold=0.3
        )

        assert result.has_context
        top = result.chunks[0]
        assert "Paris" in top.text
        assert top.source == "doc.txt"
        assert top.document_id
        assert top.score >= 0.3
        scores = [chunk.score for chunk in result.chunks]
        assert scores == sorted(scores, reverse=True)
    finally:
        await delete_collection(qdrant, collection)


async def test_search_excludes_other_tenants_points(
    qdrant: QdrantClientT, embedder: EmbeddingClient
) -> None:
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    collection = collection_name_for_tenant(tenant_a)
    try:
        await ensure_collection(qdrant, collection, embedder.dimension)
        text = "tenant data about invoices and billing"
        vector = await embedder.embed(text)
        await upsert_chunks(
            qdrant,
            collection,
            [
                ChunkPoint(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "tenant_id": str(tenant_a),
                        "document_id": str(uuid.uuid4()),
                        "source": "a.txt",
                        "chunk_index": 0,
                        "text": text,
                    },
                ),
                # A point WRONGLY tagged as tenant B, sitting in tenant A's collection.
                ChunkPoint(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "tenant_id": str(tenant_b),
                        "document_id": str(uuid.uuid4()),
                        "source": "b.txt",
                        "chunk_index": 0,
                        "text": "tenant B leaked secret",
                    },
                ),
            ],
        )

        # Accept everything by score; only the tenant_id filter should exclude anything.
        result = await retrieve_relevant_chunks(
            qdrant, embedder, tenant_a, "invoices and billing", top_k=10, score_threshold=-1.0
        )

        sources = {chunk.source for chunk in result.chunks}
        assert "a.txt" in sources
        assert "b.txt" not in sources
        assert all(chunk.text != "tenant B leaked secret" for chunk in result.chunks)
    finally:
        await delete_collection(qdrant, collection)


async def test_below_threshold_returns_no_context(
    qdrant: QdrantClientT, embedder: EmbeddingClient
) -> None:
    tenant = uuid.uuid4()
    collection = collection_name_for_tenant(tenant)
    try:
        await _seed(qdrant, embedder, tenant, ["Some unrelated content about gardening tools."])

        result = await retrieve_relevant_chunks(
            qdrant,
            embedder,
            tenant,
            "quantum chromodynamics lagrangian",
            top_k=5,
            score_threshold=0.99,
        )

        assert result.has_context is False
        assert result.chunks == []
    finally:
        await delete_collection(qdrant, collection)


async def test_empty_question_returns_no_context(
    qdrant: QdrantClientT, embedder: EmbeddingClient
) -> None:
    result = await retrieve_relevant_chunks(
        qdrant, embedder, uuid.uuid4(), "   ", top_k=5, score_threshold=0.5
    )

    assert result.has_context is False


async def test_unknown_tenant_returns_no_context(
    qdrant: QdrantClientT, embedder: EmbeddingClient
) -> None:
    result = await retrieve_relevant_chunks(
        qdrant, embedder, uuid.uuid4(), "anything", top_k=5, score_threshold=0.5
    )

    assert result.has_context is False
    assert result.chunks == []


async def test_dimension_mismatch_raises(qdrant: QdrantClientT, embedder: EmbeddingClient) -> None:
    tenant = uuid.uuid4()
    collection = collection_name_for_tenant(tenant)
    try:
        # Collection built with a different dimension than the model produces.
        await ensure_collection(qdrant, collection, embedder.dimension + 1)

        with pytest.raises(EmbeddingDimensionError):
            await retrieve_relevant_chunks(
                qdrant, embedder, tenant, "question", top_k=5, score_threshold=0.5
            )
    finally:
        await delete_collection(qdrant, collection)

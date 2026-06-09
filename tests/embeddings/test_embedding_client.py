import pytest

from app.config import get_settings
from app.embeddings.client import EmbeddingClient, EmbeddingDimensionError


@pytest.fixture(scope="module")
def embedding_client() -> EmbeddingClient:
    settings = get_settings()
    return EmbeddingClient(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )


async def test_embed_returns_configured_dimension(embedding_client: EmbeddingClient) -> None:
    vector = await embedding_client.embed("hello world")

    assert len(vector) == embedding_client.dimension
    assert all(isinstance(x, float) for x in vector)


async def test_embed_batch_returns_one_vector_per_input(
    embedding_client: EmbeddingClient,
) -> None:
    vectors = await embedding_client.embed_batch(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == embedding_client.dimension for v in vectors)


async def test_embed_batch_empty_returns_empty(embedding_client: EmbeddingClient) -> None:
    assert await embedding_client.embed_batch([]) == []


async def test_embed_is_deterministic(embedding_client: EmbeddingClient) -> None:
    first = await embedding_client.embed("same text")
    second = await embedding_client.embed("same text")

    assert first == second


async def test_dimension_mismatch_raises_loudly() -> None:
    settings = get_settings()
    misconfigured = EmbeddingClient(model_name=settings.embedding_model, dimension=999)

    with pytest.raises(EmbeddingDimensionError):
        await misconfigured.embed("anything")


def test_model_name_and_dimension_come_from_settings(
    embedding_client: EmbeddingClient,
) -> None:
    settings = get_settings()

    assert embedding_client.model_name == settings.embedding_model
    assert embedding_client.dimension == settings.embedding_dimension

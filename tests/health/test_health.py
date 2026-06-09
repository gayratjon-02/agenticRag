from fastapi import FastAPI
from httpx import AsyncClient

from app.clients.qdrant import qdrant_health
from app.clients.redis import redis_health
from app.common import Status
from app.db.session import postgres_health


async def _ok() -> Status:
    return Status.OK


async def _down() -> Status:
    return Status.DOWN


async def test_health_all_ok(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[postgres_health] = _ok
    app.dependency_overrides[redis_health] = _ok
    app.dependency_overrides[qdrant_health] = _ok

    resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "postgres": "ok",
        "redis": "ok",
        "qdrant": "ok",
    }


async def test_health_degraded_when_one_down(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[postgres_health] = _ok
    app.dependency_overrides[redis_health] = _ok
    app.dependency_overrides[qdrant_health] = _down

    resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] == "down"
    assert body["postgres"] == "ok"
    assert body["redis"] == "ok"


class _RaisingQdrant:
    async def get_collections(self) -> None:
        raise RuntimeError("qdrant unreachable")


async def test_health_handles_provider_exception(app: FastAPI, client: AsyncClient) -> None:
    # The real qdrant_health probe runs against a client that raises:
    # the endpoint must report "down" with a controlled 503, never a 500 stack trace.
    app.dependency_overrides[postgres_health] = _ok
    app.dependency_overrides[redis_health] = _ok
    app.state.qdrant = _RaisingQdrant()

    resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] == "down"

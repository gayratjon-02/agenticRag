import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.clients.redis import create_redis_client
from app.config import get_settings

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def rate_limited_client(integration_app: FastAPI) -> AsyncIterator[AsyncClient]:
    # Wire a real Redis client so the rate limiter is actually enforced.
    redis = create_redis_client(get_settings())
    integration_app.state.redis = redis
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await redis.aclose()


async def _create_tenant(client: AsyncClient) -> str:
    body = (await client.post("/tenants", json={"name": "Acme"})).json()
    api_key: str = body["api_key"]
    return api_key


async def test_requests_over_limit_return_429(
    rate_limited_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 3)
    key = await _create_tenant(rate_limited_client)
    headers = {"X-API-Key": key}

    statuses = [
        (await rate_limited_client.get(f"/documents/{uuid.uuid4()}", headers=headers)).status_code
        for _ in range(4)
    ]

    # First 3 pass the limiter (404 = not found), the 4th is rate limited.
    assert statuses[:3] == [404, 404, 404]
    assert statuses[3] == 429

import uuid
from typing import Any

import pytest
from httpx import AsyncClient

from app.documents.tasks import ingest_document_task

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def no_real_celery(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Capture enqueue calls instead of sending tasks to a real broker."""
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(ingest_document_task, "delay", lambda *args: calls.append(args))
    return calls


async def _create_tenant(client: AsyncClient, name: str = "Acme") -> str:
    body = (await client.post("/tenants", json={"name": name})).json()
    api_key: str = body["api_key"]
    return api_key


async def test_upload_returns_queued_and_enqueues(
    integration_client: AsyncClient, no_real_celery: list[tuple[Any, ...]]
) -> None:
    key = await _create_tenant(integration_client)

    resp = await integration_client.post(
        "/documents",
        headers={"X-API-Key": key},
        files={"file": ("notes.txt", b"hello world content", "text/plain")},
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["source"] == "notes.txt"
    assert body["chunk_count"] == 0
    assert len(no_real_celery) == 1  # ingestion enqueued exactly once


async def test_upload_requires_api_key(integration_client: AsyncClient) -> None:
    resp = await integration_client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert resp.status_code == 401


async def test_upload_rejects_empty_document(integration_client: AsyncClient) -> None:
    key = await _create_tenant(integration_client)

    resp = await integration_client.post(
        "/documents",
        headers={"X-API-Key": key},
        files={"file": ("empty.txt", b"   ", "text/plain")},
    )

    assert resp.status_code == 400


async def test_upload_rejects_non_utf8(integration_client: AsyncClient) -> None:
    key = await _create_tenant(integration_client)

    resp = await integration_client.post(
        "/documents",
        headers={"X-API-Key": key},
        files={"file": ("bin.dat", b"\xff\xfe\x00\x01", "application/octet-stream")},
    )

    assert resp.status_code == 400


async def test_get_status_returns_document(integration_client: AsyncClient) -> None:
    key = await _create_tenant(integration_client)
    created = (
        await integration_client.post(
            "/documents",
            headers={"X-API-Key": key},
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
    ).json()

    resp = await integration_client.get(f"/documents/{created['id']}", headers={"X-API-Key": key})

    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


async def test_get_status_requires_api_key(integration_client: AsyncClient) -> None:
    resp = await integration_client.get(f"/documents/{uuid.uuid4()}")

    assert resp.status_code == 401


async def test_get_status_rejects_bad_uuid(integration_client: AsyncClient) -> None:
    key = await _create_tenant(integration_client)

    resp = await integration_client.get("/documents/not-a-uuid", headers={"X-API-Key": key})

    assert resp.status_code == 422


async def test_get_status_unknown_returns_404(integration_client: AsyncClient) -> None:
    key = await _create_tenant(integration_client)

    resp = await integration_client.get(f"/documents/{uuid.uuid4()}", headers={"X-API-Key": key})

    assert resp.status_code == 404


async def test_get_status_isolated_across_tenants(integration_client: AsyncClient) -> None:
    key_a = await _create_tenant(integration_client, "A")
    key_b = await _create_tenant(integration_client, "B")
    created = (
        await integration_client.post(
            "/documents",
            headers={"X-API-Key": key_a},
            files={"file": ("a.txt", b"tenant A secret", "text/plain")},
        )
    ).json()

    # Tenant B must never see tenant A's document.
    resp = await integration_client.get(f"/documents/{created['id']}", headers={"X-API-Key": key_b})

    assert resp.status_code == 404

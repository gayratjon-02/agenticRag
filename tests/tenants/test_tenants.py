import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_create_tenant_returns_id_and_api_key(integration_client: AsyncClient) -> None:
    resp = await integration_client.post("/tenants", json={"name": "Acme"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Acme"
    assert "id" in body
    assert "created_at" in body
    assert body["api_key"].startswith("rag_")


async def test_create_tenant_rejects_empty_name(integration_client: AsyncClient) -> None:
    resp = await integration_client.post("/tenants", json={"name": ""})

    assert resp.status_code == 422


async def test_me_requires_api_key(integration_client: AsyncClient) -> None:
    resp = await integration_client.get("/tenants/me")

    assert resp.status_code == 401


async def test_me_rejects_invalid_api_key(integration_client: AsyncClient) -> None:
    resp = await integration_client.get("/tenants/me", headers={"X-API-Key": "rag_invalid"})

    assert resp.status_code == 401


async def test_me_returns_current_tenant(integration_client: AsyncClient) -> None:
    created = (await integration_client.post("/tenants", json={"name": "Acme"})).json()

    resp = await integration_client.get("/tenants/me", headers={"X-API-Key": created["api_key"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == created["id"]
    assert body["name"] == "Acme"
    assert "api_key" not in body  # /me must never leak the key


async def test_api_key_resolves_to_its_own_tenant(integration_client: AsyncClient) -> None:
    a = (await integration_client.post("/tenants", json={"name": "A"})).json()
    b = (await integration_client.post("/tenants", json={"name": "B"})).json()

    resp_a = await integration_client.get("/tenants/me", headers={"X-API-Key": a["api_key"]})
    resp_b = await integration_client.get("/tenants/me", headers={"X-API-Key": b["api_key"]})

    assert a["id"] != b["id"]
    assert resp_a.json()["id"] == a["id"]
    assert resp_b.json()["id"] == b["id"]

from httpx import AsyncClient


async def test_widget_is_served(client: AsyncClient) -> None:
    resp = await client.get("/widget/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Agentic RAG" in resp.text

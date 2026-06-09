import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent import service as agent_service
from app.chat.models import ChatLog
from app.clients.claude import get_claude_client
from app.clients.qdrant import get_qdrant
from app.embeddings.client import get_embedding_client
from app.retrieval.schemas import RetrievalResult, RetrievedChunk

pytestmark = pytest.mark.integration


class FakeClaude:
    """Stand-in for ClaudeClient that records calls and never hits the API."""

    def __init__(self, reply: str = "Paris is the capital of France.") -> None:
        self.reply = reply
        self.calls = 0

    async def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _result_with_context() -> RetrievalResult:
    return RetrievalResult(
        has_context=True,
        chunks=[
            RetrievedChunk(
                text="The capital of France is Paris.",
                source="geo.txt",
                document_id=str(uuid.uuid4()),
                chunk_index=0,
                score=0.9,
            )
        ],
    )


def _patch_retrieval(monkeypatch: pytest.MonkeyPatch, result: RetrievalResult) -> None:
    async def _fake(
        qdrant: object,
        embedder: object,
        tenant_id: uuid.UUID,
        question: str,
        top_k: int,
        score_threshold: float,
    ) -> RetrievalResult:
        return result

    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", _fake)


def _wire(app: FastAPI, claude: FakeClaude) -> None:
    app.dependency_overrides[get_claude_client] = lambda: claude
    app.dependency_overrides[get_qdrant] = lambda: None
    app.dependency_overrides[get_embedding_client] = lambda: None


async def _create_tenant(client: AsyncClient, name: str = "Acme") -> str:
    body = (await client.post("/tenants", json={"name": name})).json()
    api_key: str = body["api_key"]
    return api_key


async def test_chat_returns_grounded_answer_with_sources(
    integration_app: FastAPI,
    integration_client: AsyncClient,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claude = FakeClaude("Paris is the capital of France.")
    _wire(integration_app, claude)
    _patch_retrieval(monkeypatch, _result_with_context())
    key = await _create_tenant(integration_client)

    resp = await integration_client.post(
        "/chat",
        headers={"X-API-Key": key},
        json={"question": "What is the capital of France?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["escalated"] is False
    assert body["answer"] == "Paris is the capital of France."
    assert len(body["sources"]) == 1
    assert claude.calls == 1

    async with db_sessionmaker() as session:
        logs = (await session.execute(select(ChatLog))).scalars().all()
    assert len(logs) == 1
    assert logs[0].grounded is True
    assert logs[0].question == "What is the capital of France?"


async def test_chat_no_context_escalates_without_calling_claude(
    integration_app: FastAPI,
    integration_client: AsyncClient,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claude = FakeClaude()
    _wire(integration_app, claude)
    _patch_retrieval(monkeypatch, RetrievalResult(has_context=False, chunks=[]))
    key = await _create_tenant(integration_client)

    resp = await integration_client.post(
        "/chat",
        headers={"X-API-Key": key},
        json={"question": "something not in the documents"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False
    assert body["escalated"] is True
    assert body["sources"] == []
    assert "don't have that information" in body["answer"]
    assert claude.calls == 0  # Claude must NOT be called without context

    async with db_sessionmaker() as session:
        logs = (await session.execute(select(ChatLog))).scalars().all()
    assert len(logs) == 1
    assert logs[0].grounded is False


async def test_chat_logs_session_id(
    integration_app: FastAPI,
    integration_client: AsyncClient,
    db_sessionmaker: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(integration_app, FakeClaude())
    _patch_retrieval(monkeypatch, _result_with_context())
    key = await _create_tenant(integration_client)
    session_id = str(uuid.uuid4())

    resp = await integration_client.post(
        "/chat",
        headers={"X-API-Key": key},
        json={"question": "What is the capital of France?", "session_id": session_id},
    )

    assert resp.status_code == 200
    async with db_sessionmaker() as session:
        logs = (await session.execute(select(ChatLog))).scalars().all()
    assert len(logs) == 1
    assert str(logs[0].session_id) == session_id


async def test_chat_requires_api_key(
    integration_app: FastAPI,
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(integration_app, FakeClaude())
    _patch_retrieval(monkeypatch, _result_with_context())

    resp = await integration_client.post("/chat", json={"question": "hi"})

    assert resp.status_code == 401


async def test_chat_rejects_empty_question(
    integration_app: FastAPI,
    integration_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(integration_app, FakeClaude())
    _patch_retrieval(monkeypatch, _result_with_context())
    key = await _create_tenant(integration_client)

    resp = await integration_client.post("/chat", headers={"X-API-Key": key}, json={"question": ""})

    assert resp.status_code == 422

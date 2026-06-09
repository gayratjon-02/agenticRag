import uuid
from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent import service as agent_service
from app.agent.schemas import AgentAction
from app.agent.service import run_agent
from app.chat.models import ChatLog
from app.chat.schemas import ChatResponse
from app.clients.claude import ClaudeClient
from app.clients.qdrant import QdrantClientT
from app.embeddings.client import EmbeddingClient
from app.retrieval.schemas import RetrievalResult, RetrievedChunk
from app.tenants.models import Tenant
from app.tenants.security import generate_api_key, hash_api_key

pytestmark = pytest.mark.integration

QUESTION = "What is the annual revenue of Acme Corporation?"


class FakeClaude:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return "generated answer"


class ScriptedRetrieval:
    """Returns a scripted result per call and counts how many times it ran."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def __call__(
        self,
        qdrant: object,
        embedder: object,
        tenant_id: uuid.UUID,
        question: str,
        top_k: int,
        score_threshold: float,
    ) -> RetrievalResult:
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return RetrievalResult(has_context=False, chunks=[])


def _with_context() -> RetrievalResult:
    return RetrievalResult(
        has_context=True,
        chunks=[
            RetrievedChunk(
                text="Acme revenue was 10M.",
                source="acme.txt",
                document_id=str(uuid.uuid4()),
                chunk_index=0,
                score=0.8,
            )
        ],
    )


def _no_context() -> RetrievalResult:
    return RetrievalResult(has_context=False, chunks=[])


async def _make_tenant(sessionmaker: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    async with sessionmaker() as session:
        tenant = Tenant(name="Acme", api_key_hash=hash_api_key(generate_api_key()))
        session.add(tenant)
        await session.commit()
        await session.refresh(tenant)
        return tenant.id


async def _run(
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant_id: uuid.UUID,
    claude: FakeClaude,
) -> ChatResponse:
    async with sessionmaker() as session:
        return await run_agent(
            session=session,
            qdrant=cast(QdrantClientT, None),
            embedder=cast(EmbeddingClient, None),
            claude=cast(ClaudeClient, claude),
            tenant_id=tenant_id,
            question=QUESTION,
            top_k=5,
            score_threshold=0.5,
        )


async def test_answers_on_first_retrieval(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = ScriptedRetrieval([_with_context()])
    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", retrieval)
    tenant_id = await _make_tenant(db_sessionmaker)
    claude = FakeClaude()

    resp = await _run(db_sessionmaker, tenant_id, claude)

    assert resp.grounded is True
    assert resp.escalated is False
    assert [step.action for step in resp.decisions] == [AgentAction.ANSWER]
    assert retrieval.calls == 1
    assert claude.calls == 1


async def test_requeries_then_answers(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = ScriptedRetrieval([_no_context(), _with_context()])
    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", retrieval)
    tenant_id = await _make_tenant(db_sessionmaker)
    claude = FakeClaude()

    resp = await _run(db_sessionmaker, tenant_id, claude)

    assert resp.grounded is True
    assert resp.escalated is False
    assert [step.action for step in resp.decisions] == [AgentAction.REQUERY, AgentAction.ANSWER]
    assert retrieval.calls == 2
    assert claude.calls == 1


async def test_escalates_after_one_requery(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = ScriptedRetrieval([_no_context(), _no_context()])
    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", retrieval)
    tenant_id = await _make_tenant(db_sessionmaker)
    claude = FakeClaude()

    resp = await _run(db_sessionmaker, tenant_id, claude)

    assert resp.grounded is False
    assert resp.escalated is True
    assert resp.sources == []
    assert [step.action for step in resp.decisions] == [AgentAction.REQUERY, AgentAction.ESCALATE]
    assert retrieval.calls == 2  # re-query is capped at one (no loop)
    assert claude.calls == 0  # no Claude call without context


async def test_every_decision_has_a_reason(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = ScriptedRetrieval([_no_context(), _no_context()])
    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", retrieval)
    tenant_id = await _make_tenant(db_sessionmaker)

    resp = await _run(db_sessionmaker, tenant_id, FakeClaude())

    assert all(step.reason.strip() for step in resp.decisions)


async def test_interaction_is_logged(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    retrieval = ScriptedRetrieval([_with_context()])
    monkeypatch.setattr(agent_service, "retrieve_relevant_chunks", retrieval)
    tenant_id = await _make_tenant(db_sessionmaker)

    await _run(db_sessionmaker, tenant_id, FakeClaude())

    async with db_sessionmaker() as session:
        logs = (await session.execute(select(ChatLog))).scalars().all()
    assert len(logs) == 1
    assert logs[0].tenant_id == tenant_id

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Connection settings for tests. Match docker-compose host ports. Unit tests never
# open these; integration tests (marked) use them against the running stack.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5440/rag")
os.environ.setdefault("REDIS_URL", "redis://localhost:6390/0")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import create_app  # noqa: E402
from app.tenants import models as _tenant_models  # noqa: E402, F401  (register tables)


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- integration fixtures (require a running Postgres) ---


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_sessionmaker(
    db_engine: AsyncEngine,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # Start each test from a clean slate.
    async with db_engine.begin() as conn:
        await conn.execute(text("TRUNCATE tenants RESTART IDENTITY CASCADE"))
    yield async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
def integration_app(app: FastAPI, db_sessionmaker: async_sessionmaker[AsyncSession]) -> FastAPI:
    async def _get_db() -> AsyncIterator[AsyncSession]:
        async with db_sessionmaker() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    return app


@pytest_asyncio.fixture
async def integration_client(integration_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=integration_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

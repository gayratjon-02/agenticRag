import logging
from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.common import Status
from app.config import Settings

logger = logging.getLogger(__name__)


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.database_url, pool_pre_ping=True, future=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a request-scoped database session."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        yield session


async def postgres_health(request: Request) -> Status:
    """Probe Postgres connectivity. Returns DOWN instead of raising on failure."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    try:
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        return Status.OK
    except Exception:
        logger.exception("Postgres health check failed")
        return Status.DOWN

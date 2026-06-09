import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.clients.qdrant import create_qdrant_client
from app.clients.redis import create_redis_client
from app.config import get_settings
from app.db.session import create_engine, create_sessionmaker
from app.embeddings.client import create_embedding_client
from app.health.router import router as health_router
from app.tenants.router import router as tenants_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open shared clients on startup, dispose them on shutdown."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    engine = create_engine(settings)
    app.state.engine = engine
    app.state.sessionmaker = create_sessionmaker(engine)
    app.state.redis = create_redis_client(settings)
    app.state.qdrant = create_qdrant_client(settings)
    # Loading the embedding model is CPU-bound; keep it off the event loop.
    app.state.embedding = await asyncio.to_thread(create_embedding_client, settings)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await app.state.qdrant.close()
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="Agentic RAG Chatbot", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(tenants_router)
    return app


app = create_app()

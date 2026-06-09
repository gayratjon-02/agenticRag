from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.clients.qdrant import qdrant_health
from app.clients.redis import redis_health
from app.common import Status
from app.db.session import postgres_health
from app.health.schemas import HealthResponse
from app.health.service import build_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    postgres: Annotated[Status, Depends(postgres_health)],
    redis: Annotated[Status, Depends(redis_health)],
    qdrant: Annotated[Status, Depends(qdrant_health)],
) -> HealthResponse:
    result = build_health(postgres=postgres, redis=redis, qdrant=qdrant)
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result

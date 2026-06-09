from typing import Literal

from app.common import Status
from app.health.schemas import HealthResponse


def build_health(postgres: Status, redis: Status, qdrant: Status) -> HealthResponse:
    """Aggregate per-dependency statuses into an overall health response."""
    all_ok = all(s is Status.OK for s in (postgres, redis, qdrant))
    overall: Literal["ok", "degraded"] = "ok" if all_ok else "degraded"
    return HealthResponse(status=overall, postgres=postgres, redis=redis, qdrant=qdrant)

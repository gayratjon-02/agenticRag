from typing import Literal

from pydantic import BaseModel

from app.common import Status


class HealthResponse(BaseModel):
    """Aggregate health of the service and its downstream dependencies."""

    status: Literal["ok", "degraded"]
    postgres: Status
    redis: Status
    qdrant: Status

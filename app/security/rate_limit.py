import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.clients.redis import incr_with_ttl
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60


async def rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Fixed-window per-client rate limit, backed by Redis.

    Keyed by API key when present, otherwise by client IP. Fails open if Redis is
    not available (e.g. during unit tests) so it never takes the whole app down.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return

    api_key = request.headers.get("x-api-key")
    client_id = api_key or (request.client.host if request.client else "anonymous")
    count = await incr_with_ttl(redis, f"ratelimit:{client_id}", _WINDOW_SECONDS)
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
        )

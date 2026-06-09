import logging

from fastapi import Request
from redis.asyncio import Redis

from app.common import Status
from app.config import Settings

logger = logging.getLogger(__name__)


def create_redis_client(settings: Settings) -> Redis:
    """Create the async Redis client. Only place the redis SDK is constructed."""
    client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return client


async def incr_with_ttl(redis: Redis, key: str, ttl_seconds: int) -> int:
    """Atomically increment a counter, setting its TTL on first use. Returns the count."""
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, ttl_seconds)
    return int(count)


async def redis_health(request: Request) -> Status:
    """Probe Redis connectivity. Returns DOWN instead of raising on failure."""
    redis: Redis = request.app.state.redis
    try:
        await redis.ping()
        return Status.OK
    except Exception:
        logger.exception("Redis health check failed")
        return Status.DOWN

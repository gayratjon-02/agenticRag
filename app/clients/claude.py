import logging

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from fastapi import Request

from app.config import Settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """The single wrapper around the Anthropic SDK. Only place it is imported."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def generate(self, system: str, user: str) -> str:
        """Generate an answer. `system` carries the grounding instructions."""
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if isinstance(block, TextBlock))

    async def close(self) -> None:
        await self._client.close()


def create_claude_client(settings: Settings) -> ClaudeClient:
    return ClaudeClient(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
    )


def get_claude_client(request: Request) -> ClaudeClient:
    """FastAPI dependency that returns the shared Claude client."""
    client: ClaudeClient = request.app.state.claude
    return client

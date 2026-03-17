"""LLM inference gateway protocol."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMRequest:
    """Model-agnostic LLM inference request."""

    model: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = 200
    temperature: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMResponse:
    """Model-agnostic LLM inference response."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClientPort(Protocol):
    """Gateway for LLM inference calls."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the response."""
        ...

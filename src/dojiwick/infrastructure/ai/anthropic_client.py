"""Anthropic SDK wrapper with lazy import, retry, and timeout."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.llm_client import LLMRequest, LLMResponse

if TYPE_CHECKING:
    import anthropic

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = (429, 500, 529)


@dataclass(slots=True)
class AnthropicLLMClient:
    """LLMClientPort backed by the Anthropic SDK (lazy-imported)."""

    clock: ClockPort
    api_key: str
    max_retries: int = 3
    timeout_sec: float = 10.0
    _client: anthropic.AsyncAnthropic | None = None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request via the Anthropic API."""
        client = self._get_client()

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                start_ns = self.clock.monotonic_ns()
                response = await asyncio.wait_for(
                    self._call_api(client, request),
                    timeout=self.timeout_sec,
                )
                elapsed_ms = (self.clock.monotonic_ns() - start_ns) / 1_000_000

                return LLMResponse(
                    content=self._extract_text(response),
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=elapsed_ms,
                )
            except TimeoutError:
                last_exc = TimeoutError(f"LLM call timed out after {self.timeout_sec}s")
                log.warning("LLM timeout attempt=%d/%d", attempt + 1, self.max_retries + 1)
            except Exception as exc:
                if _is_retryable(exc):
                    last_exc = exc
                    delay = min(2**attempt, 30)
                    log.warning(
                        "LLM retryable error attempt=%d/%d delay=%ds: %s", attempt + 1, self.max_retries + 1, delay, exc
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        raise last_exc if last_exc is not None else OSError("LLM call failed")

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        """Extract text from the first content block."""
        import anthropic as _anthropic

        block = response.content[0]
        if isinstance(block, _anthropic.types.TextBlock):
            return block.text
        raise TypeError(f"Expected TextBlock, got {type(block).__name__}")

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            import anthropic as _anthropic

            self._client = _anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def _call_api(self, client: anthropic.AsyncAnthropic, request: LLMRequest) -> anthropic.types.Message:
        return await client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.user_prompt}],
        )


def _is_retryable(exc: Exception) -> bool:
    """Check if an Anthropic API error is retryable."""
    status = getattr(exc, "status_code", None)
    if status is not None and status in _RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, OSError | ConnectionError):
        return True
    return False

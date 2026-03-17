"""Shared constants and helpers for LLM service modules."""

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.llm_client import LLMClientPort, LLMRequest, LLMResponse
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort

MAX_RESPONSE_LENGTH = 2000


async def timed_llm_call(
    client: LLMClientPort,
    request: LLMRequest,
    metrics: MetricsSinkPort | None,
    clock: ClockPort,
) -> LLMResponse:
    """Call LLM and observe latency when metrics sink is available."""
    t0 = clock.monotonic_ns()
    response = await client.complete(request)
    if metrics is not None:
        metrics.observe("ai_response_latency_seconds", (clock.monotonic_ns() - t0) / 1_000_000_000)
    return response

"""LLM client test doubles."""

from dojiwick.domain.contracts.gateways.llm_client import LLMRequest, LLMResponse


class FixedLLMClient:
    """Returns a fixed response and records all requests."""

    def __init__(self, content: str = '{"approved": true, "reason": "approved"}') -> None:
        self._content = content
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            content=self._content,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
        )


class FailingLLMClient:
    """Raises a configurable exception on every call."""

    def __init__(self, raises: type[Exception] = OSError) -> None:
        self._raises = raises
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        raise self._raises("llm_client_error")

"""Explicit no-op metrics adapter."""


class NullMetrics:
    """Discards all metric calls."""

    def increment(self, name: str, value: int = 1) -> None:
        del name, value

    def observe(self, name: str, value: float) -> None:
        del name, value

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        del name, value, tags

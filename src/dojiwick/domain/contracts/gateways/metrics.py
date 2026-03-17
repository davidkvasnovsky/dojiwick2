"""Metrics sink protocol."""

from typing import Protocol


class MetricsSinkPort(Protocol):
    """Observability counter and histogram sink."""

    def increment(self, name: str, value: int = 1) -> None:
        """Increment counter metric."""
        ...

    def observe(self, name: str, value: float) -> None:
        """Observe numeric metric."""
        ...

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        """Set gauge metric."""
        ...

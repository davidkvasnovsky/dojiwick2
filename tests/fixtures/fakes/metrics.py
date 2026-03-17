"""Metrics test doubles."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class CapturingMetrics:
    """Captures all metric calls for assertion."""

    counters: dict[str, int] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)
    gauges: dict[str, tuple[float, dict[str, str]]] = field(default_factory=dict)

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        self.observations.setdefault(name, []).append(value)

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        self.gauges[name] = (value, tags or {})


class NullMetrics:
    """Discards all metric calls."""

    def increment(self, name: str, value: int = 1) -> None:
        del name, value

    def observe(self, name: str, value: float) -> None:
        del name, value

    def gauge(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        del name, value, tags

"""Signal repository test doubles."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.signal import Signal


@dataclass(slots=True)
class InMemorySignalRepo:
    """In-memory signal store for tests."""

    _signals: list[tuple[int, Signal]] = field(default_factory=list)
    _next_id: int = 1

    async def record_signal(self, signal: Signal, *, venue: str, product: str) -> int:
        del venue, product
        signal_id = self._next_id
        self._next_id += 1
        self._signals.append((signal_id, signal))
        return signal_id

    async def get_signals_for_tick(self, pair: str, start: datetime, end: datetime) -> tuple[Signal, ...]:
        return tuple(
            signal
            for _, signal in self._signals
            if signal.pair == pair and signal.detected_at is not None and start <= signal.detected_at <= end
        )


class FailingSignalRepo:
    """Raises on all operations."""

    async def record_signal(self, signal: Signal, *, venue: str, product: str) -> int:
        del signal, venue, product
        raise RuntimeError("signal repo failure")

    async def get_signals_for_tick(self, pair: str, start: datetime, end: datetime) -> tuple[Signal, ...]:
        del pair, start, end
        raise RuntimeError("signal repo failure")

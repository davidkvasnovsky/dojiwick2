"""Signal repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.signal import Signal


class SignalRepositoryPort(Protocol):
    """Detected market event persistence."""

    async def record_signal(self, signal: Signal, *, venue: str, product: str) -> int:
        """Persist a detected signal and return its ID."""
        ...

    async def get_signals_for_tick(self, pair: str, start: datetime, end: datetime) -> tuple[Signal, ...]:
        """Return signals detected within the given time range."""
        ...

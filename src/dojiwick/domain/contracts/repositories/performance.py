"""Performance snapshot repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot


class PerformanceRepositoryPort(Protocol):
    """Periodic equity and PnL snapshot persistence."""

    async def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Persist a performance snapshot."""
        ...

    async def get_snapshots(self, start: datetime, end: datetime) -> tuple[PerformanceSnapshot, ...]:
        """Return snapshots within the given time range."""
        ...

"""Performance repository test doubles."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot


@dataclass(slots=True)
class InMemoryPerformanceRepo:
    """In-memory performance snapshot store for tests."""

    _snapshots: list[PerformanceSnapshot] = field(default_factory=list)

    async def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        self._snapshots.append(snapshot)

    async def get_snapshots(self, start: datetime, end: datetime) -> tuple[PerformanceSnapshot, ...]:
        return tuple(s for s in self._snapshots if start <= s.observed_at <= end)


class FailingPerformanceRepo:
    """Raises on all operations."""

    async def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        del snapshot
        raise RuntimeError("performance repo failure")

    async def get_snapshots(self, start: datetime, end: datetime) -> tuple[PerformanceSnapshot, ...]:
        del start, end
        raise RuntimeError("performance repo failure")

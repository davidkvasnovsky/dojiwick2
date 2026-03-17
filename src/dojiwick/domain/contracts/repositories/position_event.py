"""Position event repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord


class PositionEventRepositoryPort(Protocol):
    """Immutable position lifecycle event persistence."""

    async def record_event(self, event: PositionEventRecord) -> int:
        """Persist a position event and return the DB-assigned id."""
        ...

    async def get_events_for_leg(self, position_leg_id: int) -> tuple[PositionEventRecord, ...]:
        """Return all events for a position leg, ordered by occurred_at."""
        ...

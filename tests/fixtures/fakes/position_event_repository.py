"""Fake position event repository for tests."""

from dataclasses import dataclass, field, replace

from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord


@dataclass(slots=True)
class FakePositionEventRepo:
    """In-memory position event repository."""

    events: list[PositionEventRecord] = field(default_factory=list)
    _next_id: int = 1

    async def record_event(self, event: PositionEventRecord) -> int:
        db_id = self._next_id
        self._next_id += 1
        self.events.append(replace(event, id=db_id))
        return db_id

    async def get_events_for_leg(self, position_leg_id: int) -> tuple[PositionEventRecord, ...]:
        return tuple(e for e in self.events if e.position_leg_id == position_leg_id)

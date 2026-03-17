"""Fake order event repository for tests."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.order_event import OrderEvent


@dataclass(slots=True)
class FakeOrderEventRepository:
    """In-memory order event repository for test assertions."""

    events: list[OrderEvent] = field(default_factory=list)

    async def record_event(self, event: OrderEvent) -> None:
        self.events.append(event)

    async def get_events_for_order(self, order_id: int) -> tuple[OrderEvent, ...]:
        return tuple(e for e in self.events if e.order_id == order_id)

    async def get_events_since(self, since: datetime) -> tuple[OrderEvent, ...]:
        return tuple(e for e in self.events if e.occurred_at >= since)

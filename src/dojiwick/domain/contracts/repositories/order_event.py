"""Order event repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.order_event import OrderEvent


class OrderEventRepositoryPort(Protocol):
    """Order event persistence and retrieval."""

    async def record_event(self, event: OrderEvent) -> None:
        """Persist an order event."""
        ...

    async def get_events_for_order(self, order_id: int) -> tuple[OrderEvent, ...]:
        """Return all events belonging to an order."""
        ...

    async def get_events_since(self, since: datetime) -> tuple[OrderEvent, ...]:
        """Return all order events since the given timestamp."""
        ...

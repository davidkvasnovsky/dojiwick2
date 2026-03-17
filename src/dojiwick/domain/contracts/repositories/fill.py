"""Fill repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.order_request import Fill


class FillRepositoryPort(Protocol):
    """Individual fill event persistence."""

    async def insert_fill(self, fill: Fill) -> int | None:
        """Persist a fill event and return the DB-assigned id, or None on dedup."""
        ...

    async def get_fills_for_order(self, order_request_id: int) -> tuple[Fill, ...]:
        """Return all fills for an order request, ordered by filled_at."""
        ...

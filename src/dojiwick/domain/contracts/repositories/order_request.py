"""Order request repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.order_request import OrderRequest
from dojiwick.domain.numerics import Quantity


class OrderRequestRepositoryPort(Protocol):
    """Order request (submitted intent) persistence."""

    async def insert_request(self, request: OrderRequest) -> int:
        """Persist an order request and return the DB-assigned id."""
        ...

    async def insert_requests(self, requests: list[OrderRequest]) -> list[int]:
        """Persist order requests in one transaction; return ids in input order."""
        ...

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRequest | None:
        """Return an order request by client_order_id, or None."""
        ...

    async def advance_applied_qty(self, order_request_id: int, cumulative_qty: Quantity) -> Quantity:
        """Advance the position-applied high-water mark; return the unapplied delta (0 if none)."""
        ...

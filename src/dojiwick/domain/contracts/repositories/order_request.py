"""Order request repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.order_request import OrderRequest


class OrderRequestRepositoryPort(Protocol):
    """Order request (submitted intent) persistence."""

    async def insert_request(self, request: OrderRequest) -> int:
        """Persist an order request and return the DB-assigned id."""
        ...

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRequest | None:
        """Return an order request by client_order_id, or None."""
        ...

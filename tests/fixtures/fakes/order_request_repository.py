"""Fake order request repository for tests."""

from dataclasses import dataclass, field, replace
from decimal import Decimal

from dojiwick.domain.models.value_objects.order_request import OrderRequest


@dataclass(slots=True)
class FakeOrderRequestRepo:
    """In-memory order request repository with auto-increment id."""

    requests: list[OrderRequest] = field(default_factory=list)
    _applied: dict[int, Decimal] = field(default_factory=dict)
    _next_id: int = 1

    async def insert_request(self, request: OrderRequest) -> int:
        db_id = self._next_id
        self._next_id += 1
        self.requests.append(replace(request, id=db_id))
        return db_id

    async def insert_requests(self, requests: list[OrderRequest]) -> list[int]:
        return [await self.insert_request(request) for request in requests]

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRequest | None:
        for r in self.requests:
            if r.client_order_id == client_order_id:
                return r
        return None

    async def advance_applied_qty(self, order_request_id: int, cumulative_qty: Decimal) -> Decimal:
        applied = self._applied.get(order_request_id, Decimal(0))
        delta = cumulative_qty - applied
        if delta > 0:
            self._applied[order_request_id] = cumulative_qty
            return delta
        return Decimal(0)

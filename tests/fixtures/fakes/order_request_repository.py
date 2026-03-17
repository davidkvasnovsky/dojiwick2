"""Fake order request repository for tests."""

from dataclasses import dataclass, field, replace

from dojiwick.domain.models.value_objects.order_request import OrderRequest


@dataclass(slots=True)
class FakeOrderRequestRepo:
    """In-memory order request repository with auto-increment id."""

    requests: list[OrderRequest] = field(default_factory=list)
    _next_id: int = 1

    async def insert_request(self, request: OrderRequest) -> int:
        db_id = self._next_id
        self._next_id += 1
        self.requests.append(replace(request, id=db_id))
        return db_id

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRequest | None:
        for r in self.requests:
            if r.client_order_id == client_order_id:
                return r
        return None

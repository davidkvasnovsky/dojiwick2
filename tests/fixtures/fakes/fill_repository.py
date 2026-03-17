"""Fake fill repository for tests."""

from dataclasses import dataclass, field, replace

from dojiwick.domain.models.value_objects.order_request import Fill


@dataclass(slots=True)
class FakeFillRepo:
    """In-memory fill repository with dedup on (order_request_id, fill_id)."""

    fills: list[Fill] = field(default_factory=list)
    _next_id: int = 1

    async def insert_fill(self, fill: Fill) -> int | None:
        if fill.fill_id:
            for existing in self.fills:
                if existing.order_request_id == fill.order_request_id and existing.fill_id == fill.fill_id:
                    return None
        db_id = self._next_id
        self._next_id += 1
        self.fills.append(replace(fill, id=db_id))
        return db_id

    async def get_fills_for_order(self, order_request_id: int) -> tuple[Fill, ...]:
        return tuple(f for f in self.fills if f.order_request_id == order_request_id)

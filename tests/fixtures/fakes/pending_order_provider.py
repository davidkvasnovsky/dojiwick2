"""Fake pending order provider for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.numerics import Quantity


@dataclass(slots=True)
class FakePendingOrderProvider:
    """In-memory pending order provider for test assertions."""

    _pending: dict[tuple[str, PositionSide], Quantity] = field(default_factory=dict)

    def seed(self, symbol: str, side: PositionSide, qty: Quantity) -> None:
        self._pending[(symbol, side)] = qty

    async def get_pending_quantities(
        self,
        account: str,
    ) -> dict[tuple[str, PositionSide], Quantity]:
        del account
        return dict(self._pending)

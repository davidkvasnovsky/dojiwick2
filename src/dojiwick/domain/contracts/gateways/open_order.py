"""Open order gateway protocol."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from dojiwick.domain.enums import OrderSide, OrderStatus, PositionSide
from dojiwick.domain.numerics import Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class ExchangeOpenOrder:
    """Snapshot of an open order on the exchange."""

    exchange_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    position_side: PositionSide
    status: OrderStatus
    original_quantity: Quantity
    filled_quantity: Quantity = Decimal(0)


class OpenOrderPort(Protocol):
    """Query and cancel open orders on the exchange."""

    async def get_open_orders(self, symbol: str) -> tuple[ExchangeOpenOrder, ...]:
        """Return all open orders for a symbol."""
        ...

    async def cancel_all_open_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        ...

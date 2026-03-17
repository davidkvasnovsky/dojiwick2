"""Order lifecycle event value object."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.enums import OrderEventType
from dojiwick.domain.numerics import Money, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class OrderEvent:
    """Immutable record of a single order lifecycle event."""

    order_id: int
    event_type: OrderEventType
    occurred_at: datetime
    exchange_order_id: str = ""
    filled_quantity: Quantity = Decimal(0)
    fees_usd: Money = Decimal(0)
    fee_asset: str = ""
    native_fee_amount: Money = Decimal(0)
    realized_pnl_exchange: Money | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity must be non-negative")
        if self.fees_usd < 0:
            raise ValueError("fees_usd must be non-negative")

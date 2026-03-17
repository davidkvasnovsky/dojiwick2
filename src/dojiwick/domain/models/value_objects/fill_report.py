"""Fill report value object.

Represents the economics of a single fill event,
separate from submission acknowledgement.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class FillReport:
    """Economics of a single order fill."""

    fill_price: Price
    filled_quantity: Quantity
    fees_usd: Money = Decimal(0)
    fee_asset: str = ""
    native_fee_amount: Money = Decimal(0)
    exchange_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.fill_price <= 0:
            raise ValueError("fill_price must be positive")
        if self.filled_quantity <= 0:
            raise ValueError("filled_quantity must be positive")
        if self.fees_usd < 0:
            raise ValueError("fees_usd must be non-negative")
        if self.native_fee_amount < 0:
            raise ValueError("native_fee_amount must be non-negative")

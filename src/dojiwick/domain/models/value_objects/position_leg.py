"""Position leg value objects for the hedge-native position model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.enums import PositionEventType, PositionSide
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class PositionLeg:
    """A single position leg stored in position_legs table."""

    account: str
    instrument_id: int
    position_side: PositionSide
    quantity: Quantity = Decimal(0)
    entry_price: Price = Decimal(0)
    unrealized_pnl: Money = Decimal(0)
    leverage: int = 1
    liquidation_price: Price | None = None
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.account:
            raise ValueError("account must not be empty")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.leverage < 1:
            raise ValueError("leverage must be >= 1")


@dataclass(slots=True, frozen=True, kw_only=True)
class PositionEventRecord:
    """An immutable lifecycle event for a position leg."""

    position_leg_id: int
    event_type: PositionEventType
    quantity: Quantity
    price: Price
    realized_pnl: Money | None = None
    occurred_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")

"""Order lifecycle value objects: request, report, fill."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
    PositionSide,
    WorkingType,
)
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class OrderRequest:
    """A submitted order intent stored in order_requests table."""

    client_order_id: str
    instrument_id: int
    account: str
    venue: str
    product: str
    tick_id: str = ""
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Price | None = None
    position_side: PositionSide = PositionSide.NET
    reduce_only: bool = False
    close_position: bool = False
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    working_type: WorkingType = WorkingType.CONTRACT_PRICE
    price_protect: bool = False
    recv_window_ms: int | None = None
    created_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.client_order_id:
            raise ValueError("client_order_id must not be empty")
        if not self.account:
            raise ValueError("account must not be empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(slots=True, frozen=True, kw_only=True)
class OrderReport:
    """Exchange-acknowledged order state from order_reports table."""

    order_request_id: int
    exchange_order_id: str
    status: OrderStatus
    filled_qty: Quantity = Decimal(0)
    avg_price: Price | None = None
    cumulative_quote_qty: Quantity | None = None
    reported_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.exchange_order_id:
            raise ValueError("exchange_order_id must not be empty")
        if self.filled_qty < 0:
            raise ValueError("filled_qty must be non-negative")


@dataclass(slots=True, frozen=True, kw_only=True)
class Fill:
    """An individual fill event from the fills table."""

    order_request_id: int
    price: Price
    quantity: Quantity
    fill_id: str = ""
    commission: Money = Decimal(0)
    commission_asset: str = ""
    realized_pnl_exchange: Money | None = None
    filled_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.commission < 0:
            raise ValueError("commission must be non-negative")

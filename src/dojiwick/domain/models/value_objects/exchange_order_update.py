"""Domain value object for exchange order updates."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class ExchangeOrderUpdate:
    """All fields from an exchange ORDER_TRADE_UPDATE event."""

    exchange_order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    order_status: OrderStatus
    execution_type: str
    position_side: PositionSide
    last_filled_qty: Quantity
    last_filled_price: Price
    cumulative_filled_qty: Quantity
    avg_price: Price
    commission: Money
    commission_asset: str
    trade_id: int
    order_trade_time: datetime
    reduce_only: bool
    close_position: bool
    realized_profit: Money
    event_time: datetime
    transaction_time: datetime

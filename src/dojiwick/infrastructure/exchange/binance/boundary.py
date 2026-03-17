"""Boundary conversion utilities for Binance exchange adapter boundaries.

These converters translate between Binance REST/WS raw data (strings, floats)
and domain types (Price, Money, Quantity, enums). All exchange-specific format
knowledge is encapsulated here — the domain layer never sees raw exchange data.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from dojiwick.domain.enums import (
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
    PositionMode,
    PositionSide,
    WorkingType,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.numerics import Money, Price, Quantity, to_money, to_price, to_quantity


# Enum conversions: Binance string <-> domain enum

_BINANCE_ORDER_SIDE: dict[str, OrderSide] = {
    "BUY": OrderSide.BUY,
    "SELL": OrderSide.SELL,
}

_BINANCE_ORDER_TYPE: dict[str, OrderType] = {
    "LIMIT": OrderType.LIMIT,
    "MARKET": OrderType.MARKET,
    "STOP_MARKET": OrderType.STOP_MARKET,
    "STOP": OrderType.STOP_LIMIT,
    "TAKE_PROFIT_MARKET": OrderType.TAKE_PROFIT_MARKET,
}

_BINANCE_ORDER_STATUS: dict[str, OrderStatus] = {
    "NEW": OrderStatus.NEW,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "EXPIRED": OrderStatus.EXPIRED,
    "REJECTED": OrderStatus.REJECTED,
}

_BINANCE_POSITION_SIDE: dict[str, PositionSide] = {
    "BOTH": PositionSide.NET,
    "LONG": PositionSide.LONG,
    "SHORT": PositionSide.SHORT,
}

_BINANCE_POSITION_MODE: dict[str, PositionMode] = {
    "true": PositionMode.HEDGE,
    "false": PositionMode.ONE_WAY,
}

_BINANCE_TIF: dict[str, OrderTimeInForce] = {
    "GTC": OrderTimeInForce.GTC,
    "IOC": OrderTimeInForce.IOC,
    "FOK": OrderTimeInForce.FOK,
    "GTX": OrderTimeInForce.GTX,
}

_BINANCE_WORKING_TYPE: dict[str, WorkingType] = {
    "MARK_PRICE": WorkingType.MARK_PRICE,
    "CONTRACT_PRICE": WorkingType.CONTRACT_PRICE,
}

# Reverse maps: domain -> Binance string
_DOMAIN_TO_BINANCE_SIDE: dict[OrderSide, str] = {v: k for k, v in _BINANCE_ORDER_SIDE.items()}
_DOMAIN_TO_BINANCE_TYPE: dict[OrderType, str] = {v: k for k, v in _BINANCE_ORDER_TYPE.items()}
_DOMAIN_TO_BINANCE_POS_SIDE: dict[PositionSide, str] = {
    PositionSide.NET: "BOTH",
    PositionSide.LONG: "LONG",
    PositionSide.SHORT: "SHORT",
}
_DOMAIN_TO_BINANCE_TIF: dict[OrderTimeInForce, str] = {v: k for k, v in _BINANCE_TIF.items()}
_DOMAIN_TO_BINANCE_WORKING: dict[WorkingType, str] = {v: k for k, v in _BINANCE_WORKING_TYPE.items()}


# Inbound: Binance -> domain


def parse_order_side(raw: str) -> OrderSide:
    """Convert a Binance order side string to domain enum."""
    return _BINANCE_ORDER_SIDE[raw.upper()]


def parse_order_type(raw: str) -> OrderType:
    """Convert a Binance order type string to domain enum."""
    return _BINANCE_ORDER_TYPE[raw.upper()]


def parse_order_status(raw: str) -> OrderStatus:
    """Convert a Binance order status string to domain enum."""
    return _BINANCE_ORDER_STATUS[raw.upper()]


def parse_position_side(raw: str) -> PositionSide:
    """Convert a Binance position side string to domain enum."""
    return _BINANCE_POSITION_SIDE[raw.upper()]


def parse_position_mode(hedge_mode: str) -> PositionMode:
    """Convert Binance dualSidePosition response to domain enum."""
    return _BINANCE_POSITION_MODE[hedge_mode.lower()]


def parse_time_in_force(raw: str) -> OrderTimeInForce:
    """Convert a Binance time-in-force string to domain enum."""
    return _BINANCE_TIF[raw.upper()]


def parse_working_type(raw: str) -> WorkingType:
    """Convert a Binance working type string to domain enum."""
    return _BINANCE_WORKING_TYPE[raw.upper()]


def parse_price(raw: str) -> Price:
    """Parse a Binance price string to domain Price (Decimal)."""
    return to_price(raw)


def parse_money(raw: str) -> Money:
    """Parse a Binance money string to domain Money (Decimal)."""
    return to_money(raw)


def parse_quantity(raw: str) -> Quantity:
    """Parse a Binance quantity string to domain Quantity (Decimal)."""
    return to_quantity(raw)


def build_instrument_id(symbol: str, base_asset: str, quote_asset: str, settle_asset: str = "") -> InstrumentId:
    """Build an InstrumentId from Binance exchange info fields.

    Defaults to BINANCE / USD_C. settle_asset defaults to quote_asset if empty.
    """
    return InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol=symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
        settle_asset=settle_asset or quote_asset,
    )


# Outbound: domain -> Binance


def format_order_side(side: OrderSide) -> str:
    """Convert domain OrderSide to Binance API string."""
    return _DOMAIN_TO_BINANCE_SIDE[side]


def format_order_type(order_type: OrderType) -> str:
    """Convert domain OrderType to Binance API string."""
    return _DOMAIN_TO_BINANCE_TYPE[order_type]


def format_position_side(side: PositionSide) -> str:
    """Convert domain PositionSide to Binance API string."""
    return _DOMAIN_TO_BINANCE_POS_SIDE[side]


def format_time_in_force(tif: OrderTimeInForce) -> str:
    """Convert domain OrderTimeInForce to Binance API string."""
    return _DOMAIN_TO_BINANCE_TIF[tif]


def format_working_type(wt: WorkingType) -> str:
    """Convert domain WorkingType to Binance API string."""
    return _DOMAIN_TO_BINANCE_WORKING[wt]


def format_price(price: Price) -> str:
    """Format domain Price as Binance-compatible string (no trailing zeros)."""
    return _format_decimal(price)


def format_quantity(qty: Quantity) -> str:
    """Format domain Quantity as Binance-compatible string (no trailing zeros)."""
    return _format_decimal(qty)


def _format_decimal(value: Decimal) -> str:
    """Format a Decimal without trailing zeros, suitable for Binance API params."""
    return f"{value.normalize():f}"


# Dict field extraction helpers (for raw JSON responses)


def str_field(d: dict[str, object], key: str, default: str = "") -> str:
    """Extract a string field from a raw JSON dict."""
    return str(d.get(key, default))


def int_field(d: dict[str, object], key: str, default: int = 0) -> int:
    """Extract an int field from a raw JSON dict."""
    raw = d.get(key, default)
    return int(raw) if isinstance(raw, int | float) else default


def bool_field(d: dict[str, object], key: str, *, default: bool = False) -> bool:
    """Extract a bool field from a raw JSON dict."""
    raw = d.get(key, default)
    return bool(raw)


def ms_to_utc(ms: int) -> datetime:
    """Convert millisecond epoch timestamp to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def parse_ws_order_update(raw: dict[str, object]) -> ExchangeOrderUpdate:
    """Parse a Binance ORDER_TRADE_UPDATE WS payload into an ExchangeOrderUpdate."""
    o_raw = raw.get("o", {})
    if not isinstance(o_raw, dict):
        raise ValueError("missing 'o' sub-dict in ORDER_TRADE_UPDATE payload")
    o = cast(dict[str, object], o_raw)

    return ExchangeOrderUpdate(
        exchange_order_id=str(int_field(o, "i")),
        client_order_id=str_field(o, "c"),
        symbol=str_field(o, "s"),
        side=parse_order_side(str_field(o, "S")),
        order_type=parse_order_type(str_field(o, "o")),
        order_status=parse_order_status(str_field(o, "X")),
        execution_type=str_field(o, "x"),
        position_side=parse_position_side(str_field(o, "ps")),
        last_filled_qty=parse_quantity(str_field(o, "l", "0")),
        last_filled_price=parse_price(str_field(o, "L", "0")),
        cumulative_filled_qty=parse_quantity(str_field(o, "z", "0")),
        avg_price=parse_price(str_field(o, "ap", "0")),
        commission=parse_money(str_field(o, "n", "0")),
        commission_asset=str_field(o, "N"),
        trade_id=int_field(o, "t"),
        order_trade_time=ms_to_utc(int_field(o, "T")),
        reduce_only=bool_field(o, "R"),
        close_position=bool_field(o, "cp"),
        realized_profit=parse_money(str_field(o, "rp", "0")),
        event_time=ms_to_utc(int_field(raw, "E")),
        transaction_time=ms_to_utc(int_field(raw, "T")),
    )

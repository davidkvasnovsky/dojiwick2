"""Unit tests for Binance ORDER_TRADE_UPDATE WS parsing and stream event conversion."""

from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import OrderEventType, OrderSide, OrderStatus, OrderType, PositionSide, STATUS_TO_EVENT_TYPE
from dojiwick.infrastructure.exchange.binance.boundary import parse_ws_order_update


def _sample_ws_payload(
    *,
    status: str = "FILLED",
    execution_type: str = "TRADE",
    side: str = "BUY",
    order_type: str = "MARKET",
    position_side: str = "BOTH",
    symbol: str = "BTCUSDC",
    order_id: int = 12345,
    client_order_id: str = "my_order_1",
    last_filled_qty: str = "0.01",
    last_filled_price: str = "95000.0",
    cumulative_filled_qty: str = "0.01",
    avg_price: str = "95000.0",
    commission: str = "0.5",
    commission_asset: str = "USDC",
    trade_id: int = 9999,
    order_trade_time: int = 1700000000000,
    reduce_only: bool = False,
    close_position: bool = False,
    realized_profit: str = "0",
    event_time: int = 1700000001000,
    transaction_time: int = 1700000000500,
) -> dict[str, object]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": event_time,
        "T": transaction_time,
        "o": {
            "s": symbol,
            "c": client_order_id,
            "S": side,
            "o": order_type,
            "f": "GTC",
            "q": "0.01",
            "p": "0",
            "ap": avg_price,
            "sp": "0",
            "x": execution_type,
            "X": status,
            "i": order_id,
            "l": last_filled_qty,
            "z": cumulative_filled_qty,
            "L": last_filled_price,
            "N": commission_asset,
            "n": commission,
            "T": order_trade_time,
            "t": trade_id,
            "b": "0",
            "a": "0",
            "m": False,
            "R": reduce_only,
            "wt": "CONTRACT_PRICE",
            "ot": order_type,
            "ps": position_side,
            "cp": close_position,
            "AP": "0",
            "cr": "0",
            "rp": realized_profit,
        },
    }


def test_parse_ws_order_update_filled() -> None:
    """Parse a fully filled ORDER_TRADE_UPDATE into ExchangeOrderUpdate."""
    raw = _sample_ws_payload()
    update = parse_ws_order_update(raw)

    assert update.exchange_order_id == "12345"
    assert update.client_order_id == "my_order_1"
    assert update.symbol == "BTCUSDC"
    assert update.side is OrderSide.BUY
    assert update.order_type is OrderType.MARKET
    assert update.order_status is OrderStatus.FILLED
    assert update.execution_type == "TRADE"
    assert update.position_side is PositionSide.NET
    assert update.last_filled_qty == Decimal("0.01")
    assert update.last_filled_price == Decimal("95000.0")
    assert update.cumulative_filled_qty == Decimal("0.01")
    assert update.avg_price == Decimal("95000.0")
    assert update.commission == Decimal("0.5")
    assert update.commission_asset == "USDC"
    assert update.trade_id == 9999
    assert update.reduce_only is False
    assert update.close_position is False
    assert update.realized_profit == Decimal("0")
    assert update.event_time.tzinfo is not None
    assert update.transaction_time.tzinfo is not None
    assert update.order_trade_time.tzinfo is not None


def test_parse_ws_order_update_canceled() -> None:
    """Parse a canceled order WS event."""
    raw = _sample_ws_payload(
        status="CANCELED",
        execution_type="CANCELED",
        last_filled_qty="0",
        last_filled_price="0",
        cumulative_filled_qty="0",
        avg_price="0",
        commission="0",
        trade_id=0,
    )
    update = parse_ws_order_update(raw)

    assert update.order_status is OrderStatus.CANCELED
    assert update.execution_type == "CANCELED"
    assert update.last_filled_qty == Decimal(0)
    assert update.trade_id == 0


def test_parse_ws_order_update_partial_fill() -> None:
    """Parse a partially filled WS event."""
    raw = _sample_ws_payload(
        status="PARTIALLY_FILLED",
        last_filled_qty="0.005",
        cumulative_filled_qty="0.005",
    )
    update = parse_ws_order_update(raw)

    assert update.order_status is OrderStatus.PARTIALLY_FILLED
    assert update.last_filled_qty == Decimal("0.005")
    assert update.cumulative_filled_qty == Decimal("0.005")


def test_parse_ws_order_update_with_reduce_only() -> None:
    """Parse an event with reduce_only flag."""
    raw = _sample_ws_payload(reduce_only=True, position_side="LONG")
    update = parse_ws_order_update(raw)

    assert update.reduce_only is True
    assert update.position_side is PositionSide.LONG


def test_parse_ws_order_update_sell_side() -> None:
    """Parse a SELL-side order event."""
    raw = _sample_ws_payload(side="SELL")
    update = parse_ws_order_update(raw)

    assert update.side is OrderSide.SELL


def test_parse_ws_order_update_limit_order() -> None:
    """Parse a LIMIT order event."""
    raw = _sample_ws_payload(order_type="LIMIT")
    update = parse_ws_order_update(raw)

    assert update.order_type is OrderType.LIMIT


def test_status_to_event_type_mapping() -> None:
    """All OrderStatus values have an OrderEventType mapping."""
    assert STATUS_TO_EVENT_TYPE[OrderStatus.NEW] is OrderEventType.PLACED
    assert STATUS_TO_EVENT_TYPE[OrderStatus.PARTIALLY_FILLED] is OrderEventType.PARTIALLY_FILLED
    assert STATUS_TO_EVENT_TYPE[OrderStatus.FILLED] is OrderEventType.FILLED
    assert STATUS_TO_EVENT_TYPE[OrderStatus.CANCELED] is OrderEventType.CANCELED
    assert STATUS_TO_EVENT_TYPE[OrderStatus.EXPIRED] is OrderEventType.EXPIRED
    assert STATUS_TO_EVENT_TYPE[OrderStatus.REJECTED] is OrderEventType.REJECTED


def test_parse_ws_order_update_timestamps() -> None:
    """Timestamps are correctly converted to UTC datetimes."""
    raw = _sample_ws_payload(
        event_time=1700000001000,
        transaction_time=1700000000500,
        order_trade_time=1700000000000,
    )
    update = parse_ws_order_update(raw)

    assert update.event_time == datetime.fromtimestamp(1700000001000 / 1000.0, tz=UTC)
    assert update.transaction_time == datetime.fromtimestamp(1700000000500 / 1000.0, tz=UTC)
    assert update.order_trade_time == datetime.fromtimestamp(1700000000000 / 1000.0, tz=UTC)


def test_parse_ws_order_update_missing_o_raises() -> None:
    """Missing 'o' sub-dict raises ValueError."""
    raw: dict[str, object] = {"e": "ORDER_TRADE_UPDATE", "E": 0, "T": 0, "o": "not_a_dict"}
    try:
        parse_ws_order_update(raw)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "missing 'o'" in str(exc)

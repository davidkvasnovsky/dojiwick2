"""Unit tests for BinanceOpenOrderAdapter response parsing."""

from decimal import Decimal

from dojiwick.domain.enums import OrderSide, OrderStatus, PositionSide
from dojiwick.infrastructure.exchange.binance.open_order import parse_open_order


def test_parse_open_order_response() -> None:
    raw: dict[str, object] = {
        "orderId": 12345678,
        "clientOrderId": "dw_abc12345_def",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "positionSide": "LONG",
        "status": "NEW",
        "origQty": "0.100",
        "executedQty": "0.030",
    }

    order = parse_open_order(raw)

    assert order.exchange_order_id == "12345678"
    assert order.client_order_id == "dw_abc12345_def"
    assert order.symbol == "BTCUSDC"
    assert order.side == OrderSide.BUY
    assert order.position_side == PositionSide.LONG
    assert order.status == OrderStatus.NEW
    assert order.original_quantity == Decimal("0.100")
    assert order.filled_quantity == Decimal("0.030")


def test_empty_response() -> None:
    raw_list: list[dict[str, object]] = []
    result = tuple(parse_open_order(item) for item in raw_list)
    assert result == ()

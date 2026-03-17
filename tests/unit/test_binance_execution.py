"""Unit tests for BinanceExecutionGateway."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderSide,
    OrderTimeInForce,
    OrderType,
    PositionSide,
    SubmissionStatus,
    TradeAction,
    WorkingType,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.errors import ExchangeError, OrderNotFoundError
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.infrastructure.exchange.binance.execution import BinanceExecutionGateway


def _get_request_params(mock_client: MagicMock) -> dict[str, str]:
    return mock_client.request.call_args.kwargs.get("params") or mock_client.request.call_args[1]["params"]


def _make_gateway() -> tuple[BinanceExecutionGateway, MagicMock]:
    mock_client = MagicMock()
    mock_client.request = AsyncMock()
    mock_client.request_list = AsyncMock()
    gw = BinanceExecutionGateway.__new__(BinanceExecutionGateway)
    object.__setattr__(gw, "client", mock_client)
    return gw, mock_client


_IID = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDT",
    base_asset="BTC",
    quote_asset="USDT",
    settle_asset="USDT",
)


def _make_delta(
    *,
    order_type: OrderType = OrderType.MARKET,
    sequence: int = 0,
    price: Decimal | None = None,
    reduce_only: bool = False,
    close_position: bool = False,
    side: OrderSide = OrderSide.BUY,
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
    working_type: WorkingType = WorkingType.CONTRACT_PRICE,
) -> LegDelta:
    return LegDelta(
        instrument_id=_IID,
        target_index=0,
        position_side=PositionSide.NET,
        side=side,
        order_type=order_type,
        quantity=Decimal("0.01"),
        price=price,
        reduce_only=reduce_only,
        close_position=close_position,
        sequence=sequence,
        time_in_force=time_in_force,
        working_type=working_type,
    )


def _make_plan(*deltas: LegDelta) -> ExecutionPlan:
    return ExecutionPlan(account="test", deltas=deltas)


def _filled_response(
    *,
    avg_price: str = "42000.00",
    executed_qty: str = "0.01",
    order_id: int = 12345,
    update_time: int = 1700000000000,
) -> dict[str, object]:
    return {
        "status": "FILLED",
        "avgPrice": avg_price,
        "executedQty": executed_qty,
        "orderId": order_id,
        "updateTime": update_time,
    }


def _new_response() -> dict[str, object]:
    return {
        "status": "NEW",
        "avgPrice": "0",
        "executedQty": "0",
        "orderId": 99999,
        "updateTime": 1700000000000,
    }


# TestExecutePlan


class TestExecutePlan:
    async def test_single_market_delta_filled(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta())

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert len(receipts) == 1
        r = receipts[0]
        assert r.status == ExecutionStatus.FILLED
        assert r.fill_price == Decimal("42000.00")
        assert r.filled_quantity == Decimal("0.01")
        assert r.order_id == "12345"
        assert r.exchange_timestamp is not None

    async def test_sequence_ordering(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        d0 = _make_delta(sequence=1)
        d1 = _make_delta(sequence=0)
        plan = _make_plan(d0, d1)

        await gw.execute_plan(plan, tick_id="abc123")

        # seq=0 (d1) should be POSTed first
        calls = client.request.call_args_list
        assert len(calls) == 2
        first_params = calls[0].kwargs.get("params") or calls[0][1].get("params", {})
        # The first call should be for d1 (seq=0), so its client order ID leg_seq=0
        coid = compute_client_order_id("abc123", "BTCUSDT", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
        assert first_params["newClientOrderId"] == coid

    async def test_receipts_aligned_to_original_deltas(self) -> None:
        gw, client = _make_gateway()
        resp_a = _filled_response(order_id=111)
        resp_b = _filled_response(order_id=222)
        client.request.side_effect = [resp_b, resp_a]  # seq=0 first, then seq=1
        d0 = _make_delta(sequence=1)  # original index 0, executed second
        d1 = _make_delta(sequence=0)  # original index 1, executed first
        plan = _make_plan(d0, d1)

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        # receipt[0] corresponds to d0 (original index 0, seq=1, executed second -> resp_a)
        assert receipts[0].order_id == "111"
        # receipt[1] corresponds to d1 (original index 1, seq=0, executed first -> resp_b)
        assert receipts[1].order_id == "222"

    async def test_empty_plan_returns_empty_tuple(self) -> None:
        gw, _client = _make_gateway()
        plan = _make_plan()

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert receipts == ()

    async def test_adapter_error_returns_error_receipt(self) -> None:
        gw, client = _make_gateway()
        client.request.side_effect = ExchangeError("test error")
        plan = _make_plan(_make_delta())

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert len(receipts) == 1
        assert receipts[0].status == ExecutionStatus.ERROR

    async def test_new_status_becomes_skipped(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _new_response()
        plan = _make_plan(_make_delta())

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert len(receipts) == 1
        assert receipts[0].status == ExecutionStatus.SKIPPED
        assert "order_pending:new" in receipts[0].reason

    async def test_fail_fast_skips_remaining(self) -> None:
        gw, client = _make_gateway()
        client.request.side_effect = ExchangeError("boom")
        d0 = _make_delta(sequence=0)
        d1 = _make_delta(sequence=1)
        plan = _make_plan(d0, d1)

        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert len(receipts) == 2
        assert receipts[0].status == ExecutionStatus.ERROR
        assert receipts[1].status == ExecutionStatus.ERROR
        assert receipts[1].reason == "skipped_after_failure"


# TestBuildOrderParams (via execute_plan inspection)


class TestBuildOrderParams:
    async def test_market_order_no_price_no_tif(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta(order_type=OrderType.MARKET))

        await gw.execute_plan(plan, tick_id="t1")

        params = _get_request_params(client)
        assert "price" not in params
        assert "timeInForce" not in params
        assert params["type"] == "MARKET"

    async def test_limit_order_has_price_and_tif(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta(order_type=OrderType.LIMIT, price=Decimal("42000")))

        await gw.execute_plan(plan, tick_id="t1")

        params = _get_request_params(client)
        assert params["price"] == "42000"
        assert params["timeInForce"] == "GTC"
        assert params["type"] == "LIMIT"

    async def test_stop_market_uses_stop_price(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta(order_type=OrderType.STOP_MARKET, price=Decimal("40000")))

        await gw.execute_plan(plan, tick_id="t1")

        params = _get_request_params(client)
        assert params["stopPrice"] == "40000"
        assert "price" not in params
        assert "timeInForce" not in params
        assert params["workingType"] == "CONTRACT_PRICE"

    async def test_reduce_only_flag(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta(reduce_only=True))

        await gw.execute_plan(plan, tick_id="t1")

        params = _get_request_params(client)
        assert params["reduceOnly"] == "true"

    async def test_client_order_id_deterministic(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()
        plan = _make_plan(_make_delta())

        await gw.execute_plan(plan, tick_id="tick1234")

        params = _get_request_params(client)
        expected = compute_client_order_id("tick1234", "BTCUSDT", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
        assert params["newClientOrderId"] == expected


# TestPlaceOrder


class TestPlaceOrder:
    async def test_buy_action_sends_buy_side(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()

        await gw.place_order("BTCUSDT", TradeAction.BUY, OrderType.MARKET, Decimal("42000"), Decimal("0.01"))

        params = _get_request_params(client)
        assert params["side"] == "BUY"

    async def test_short_action_sends_sell_side(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = _filled_response()

        await gw.place_order("BTCUSDT", TradeAction.SHORT, OrderType.MARKET, Decimal("42000"), Decimal("0.01"))

        params = _get_request_params(client)
        assert params["side"] == "SELL"

    async def test_hold_action_raises_value_error(self) -> None:
        gw, _client = _make_gateway()

        with pytest.raises(ValueError, match="cannot convert"):
            await gw.place_order("BTCUSDT", TradeAction.HOLD, OrderType.MARKET, Decimal("42000"), Decimal("0.01"))


# TestCancelOrder


class TestCancelOrder:
    async def test_cancel_success_returns_cancelled(self) -> None:
        gw, client = _make_gateway()
        client.request.return_value = {
            "status": "CANCELED",
            "orderId": 12345,
            "updateTime": 1700000000000,
        }

        receipt = await gw.cancel_order("BTCUSDT", "12345")

        assert receipt.status == SubmissionStatus.CANCELLED
        assert receipt.reason == "cancel_success"
        assert receipt.order_id == "12345"

    async def test_cancel_not_found_returns_error(self) -> None:
        gw, client = _make_gateway()
        client.request.side_effect = OrderNotFoundError("not found")

        receipt = await gw.cancel_order("BTCUSDT", "99999")

        assert receipt.status == SubmissionStatus.ERROR
        assert receipt.reason == "order_not_found"

    async def test_parse_canceled_returns_cancelled(self) -> None:
        """Binance CANCELED status maps to CANCELLED via execute_plan."""
        gw, client = _make_gateway()
        client.request.return_value = {
            "status": "CANCELED",
            "avgPrice": "0",
            "executedQty": "0",
            "orderId": 99999,
            "updateTime": 1700000000000,
        }
        plan = _make_plan(_make_delta())
        receipts = await gw.execute_plan(plan, tick_id="abc123")

        assert len(receipts) == 1
        assert receipts[0].status == ExecutionStatus.CANCELLED
        assert receipts[0].reason == "order_canceled"

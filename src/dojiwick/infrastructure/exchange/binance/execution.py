"""Binance execution gateway — order placement, cancellation, and plan execution."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderSide,
    OrderStatus,
    OrderTimeInForce,
    OrderType,
    PositionSide,
    SubmissionStatus,
    TradeAction,
    WorkingType,
)
from dojiwick.domain.errors import AdapterError, OrderNotFoundError
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck
from dojiwick.domain.numerics import Price, Quantity

from .boundary import (
    format_order_side,
    format_order_type,
    format_position_side,
    format_price,
    format_quantity,
    format_time_in_force,
    format_working_type,
    int_field,
    parse_order_status,
    parse_price,
    parse_quantity,
    str_field,
)
from .http_client import BinanceHttpClient

log = logging.getLogger(__name__)

_STOP_TYPES = frozenset({OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET, OrderType.STOP_LIMIT})


def _trade_action_to_order_side(action: TradeAction) -> OrderSide:
    if action == TradeAction.BUY:
        return OrderSide.BUY
    if action == TradeAction.SHORT:
        return OrderSide.SELL
    raise ValueError(f"cannot convert {action!r} to OrderSide")


def _apply_order_type_params(
    params: dict[str, str],
    order_type: OrderType,
    price: Price | None,
    time_in_force: OrderTimeInForce,
    reduce_only: bool,
    close_position: bool,
    working_type: WorkingType,
) -> None:
    """Mutate params dict with order-type-specific fields."""
    if order_type == OrderType.LIMIT:
        if price is not None:
            params["price"] = format_price(price)
        params["timeInForce"] = format_time_in_force(time_in_force)
    elif order_type == OrderType.STOP_LIMIT:
        if price is not None:
            params["stopPrice"] = format_price(price)
            params["price"] = format_price(price)
        params["timeInForce"] = format_time_in_force(time_in_force)
    elif order_type in (OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
        if price is not None:
            params["stopPrice"] = format_price(price)
    # MARKET: no price, no timeInForce

    if reduce_only:
        params["reduceOnly"] = "true"
    if close_position:
        params["closePosition"] = "true"
    if order_type in _STOP_TYPES:
        params["workingType"] = format_working_type(working_type)


def _build_order_params(delta: LegDelta, *, tick_id: str, leg_seq: int) -> dict[str, str]:
    symbol = delta.instrument_id.symbol
    params: dict[str, str] = {
        "symbol": symbol,
        "side": format_order_side(delta.side),
        "type": format_order_type(delta.order_type),
        "positionSide": format_position_side(delta.position_side),
        "quantity": format_quantity(delta.quantity),
        "newClientOrderId": compute_client_order_id(
            tick_id, symbol, delta.side, delta.position_side, leg_seq, delta.order_type
        ),
        "newOrderRespType": "RESULT",
    }
    _apply_order_type_params(
        params,
        delta.order_type,
        delta.price,
        delta.time_in_force,
        delta.reduce_only,
        delta.close_position,
        delta.working_type,
    )
    return params


def _parse_order_response(raw: dict[str, object]) -> ExecutionReceipt:
    status = parse_order_status(str_field(raw, "status"))

    if status == OrderStatus.FILLED:
        return ExecutionReceipt(
            status=ExecutionStatus.FILLED,
            reason="filled",
            fill_price=parse_price(str_field(raw, "avgPrice")),
            filled_quantity=parse_quantity(str_field(raw, "executedQty")),
            order_id=str(int_field(raw, "orderId")),
            exchange_timestamp=datetime.fromtimestamp(int_field(raw, "updateTime") / 1000, tz=UTC),
        )

    if status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
        return ExecutionReceipt(
            status=ExecutionStatus.SKIPPED,
            reason=f"order_pending:{status.value}",
        )

    if status == OrderStatus.REJECTED:
        return ExecutionReceipt(
            status=ExecutionStatus.REJECTED,
            reason="rejected",
        )

    # CANCELED, EXPIRED
    return ExecutionReceipt(
        status=ExecutionStatus.CANCELLED,
        reason=f"order_{status.value}",
    )


def _parse_submission_ack(raw: dict[str, object]) -> SubmissionAck:
    status = parse_order_status(str_field(raw, "status"))

    if status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED):
        return SubmissionAck(
            status=SubmissionStatus.ACCEPTED,
            order_id=str(int_field(raw, "orderId")),
            exchange_timestamp=datetime.fromtimestamp(int_field(raw, "updateTime") / 1000, tz=UTC),
        )

    if status == OrderStatus.REJECTED:
        return SubmissionAck(
            status=SubmissionStatus.REJECTED,
            reason="rejected",
        )

    # CANCELED, EXPIRED
    return SubmissionAck(
        status=SubmissionStatus.CANCELLED,
        reason=f"order_{status.value}",
    )


@dataclass(slots=True)
class BinanceExecutionGateway:
    """Executes orders on the Binance Futures API."""

    client: BinanceHttpClient

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        """Execute an execution plan (leg deltas) and return receipts."""
        if not plan.deltas:
            return ()

        indexed_deltas = sorted(enumerate(plan.deltas), key=lambda p: p[1].sequence)
        receipts: list[ExecutionReceipt | None] = [None] * len(plan.deltas)

        for leg_seq, (original_index, delta) in enumerate(indexed_deltas):
            try:
                params = _build_order_params(delta, tick_id=tick_id, leg_seq=leg_seq)
                raw = await self.client.request("POST", "/fapi/v1/order", params=params, signed=True)
                receipts[original_index] = _parse_order_response(raw)
            except AdapterError as exc:
                log.error("execute_plan leg %d failed: %s", leg_seq, exc)
                receipts[original_index] = ExecutionReceipt(
                    status=ExecutionStatus.ERROR,
                    reason=str(exc),
                )
                # Fail-fast: skip remaining deltas
                for remaining_seq in range(leg_seq + 1, len(indexed_deltas)):
                    remaining_original = indexed_deltas[remaining_seq][0]
                    receipts[remaining_original] = ExecutionReceipt(
                        status=ExecutionStatus.ERROR,
                        reason="skipped_after_failure",
                    )
                break

        # Defensive: fill any remaining None slots
        return tuple(
            r if r is not None else ExecutionReceipt(status=ExecutionStatus.ERROR, reason="no_response")
            for r in receipts
        )

    async def place_order(
        self,
        pair: str,
        side: TradeAction,
        order_type: OrderType,
        price: Price,
        quantity: Quantity,
        *,
        instrument_id: InstrumentId | None = None,
        client_order_id: str = "",
        exchange_order_id: str = "",
        position_side: PositionSide = PositionSide.NET,
        reduce_only: bool = False,
        close_position: bool = False,
        working_type: WorkingType = WorkingType.CONTRACT_PRICE,
        price_protect: bool = False,
        time_in_force: OrderTimeInForce = OrderTimeInForce.GTC,
        recv_window_ms: int = 5000,
    ) -> SubmissionAck:
        """Place a single order on the exchange."""
        order_side = _trade_action_to_order_side(side)

        params: dict[str, str] = {
            "symbol": pair,
            "side": format_order_side(order_side),
            "type": format_order_type(order_type),
            "positionSide": format_position_side(position_side),
            "quantity": format_quantity(quantity),
            "newOrderRespType": "RESULT",
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        _apply_order_type_params(params, order_type, price, time_in_force, reduce_only, close_position, working_type)

        try:
            raw = await self.client.request("POST", "/fapi/v1/order", params=params, signed=True)
            return _parse_submission_ack(raw)
        except AdapterError as exc:
            return SubmissionAck(
                status=SubmissionStatus.ERROR,
                reason=str(exc),
            )

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        """Cancel an existing order on the exchange."""
        try:
            raw = await self.client.request(
                "DELETE",
                "/fapi/v1/order",
                params={"symbol": pair, "orderId": order_id},
                signed=True,
            )
            status_str = str_field(raw, "status")
            if status_str.upper() == "CANCELED":
                return SubmissionAck(
                    status=SubmissionStatus.CANCELLED,
                    reason="cancel_success",
                    order_id=order_id,
                    exchange_timestamp=datetime.fromtimestamp(int_field(raw, "updateTime") / 1000, tz=UTC),
                )
            return _parse_submission_ack(raw)
        except OrderNotFoundError:
            return SubmissionAck(
                status=SubmissionStatus.ERROR,
                reason="order_not_found",
            )
        except AdapterError as exc:
            return SubmissionAck(
                status=SubmissionStatus.ERROR,
                reason=str(exc),
            )

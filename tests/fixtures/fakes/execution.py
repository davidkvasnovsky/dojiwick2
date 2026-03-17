"""Execution gateway test doubles."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import NamedTuple

from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderTimeInForce,
    OrderType,
    PositionSide,
    SubmissionStatus,
    TradeAction,
    WorkingType,
)
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck
from dojiwick.domain.numerics import Price, Quantity


class PlanCall(NamedTuple):
    """Captured arguments from a single execute_plan invocation."""

    plan: ExecutionPlan
    tick_id: str


class DryRunGateway(ExecutionGatewayPort):
    """Fills all plan deltas."""

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        receipts: list[ExecutionReceipt] = []
        for i, delta in enumerate(plan.deltas):
            fill_price = delta.price if delta.price is not None else Decimal(1)
            coid = compute_client_order_id(
                tick_id,
                delta.instrument_id.symbol,
                delta.side,
                delta.position_side,
                i,
                delta.order_type,
            )
            receipts.append(
                ExecutionReceipt(
                    status=ExecutionStatus.FILLED,
                    reason="dry_run_plan_fill",
                    fill_price=fill_price,
                    filled_quantity=delta.quantity,
                    exchange_timestamp=datetime.now(UTC),
                    order_id=coid,
                )
            )
        return tuple(receipts)

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        return SubmissionAck(
            status=SubmissionStatus.CANCELLED,
            reason=f"dry_run_cancel pair={pair} order_id={order_id}",
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
        return SubmissionAck(
            status=SubmissionStatus.ACCEPTED,
            reason="dry_run_filled",
            exchange_timestamp=datetime.now(UTC),
        )


class RejectAllGateway(ExecutionGatewayPort):
    """Rejects all plan deltas."""

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        return tuple(ExecutionReceipt(status=ExecutionStatus.REJECTED, reason="rejected") for _ in plan.deltas)

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        del pair, order_id
        return SubmissionAck(status=SubmissionStatus.REJECTED, reason="rejected")

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
        return SubmissionAck(status=SubmissionStatus.REJECTED, reason="rejected")


class ErrorGateway(ExecutionGatewayPort):
    """Returns error status for all plan deltas."""

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        return tuple(ExecutionReceipt(status=ExecutionStatus.ERROR, reason="gateway_error") for _ in plan.deltas)

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        del pair, order_id
        return SubmissionAck(status=SubmissionStatus.ERROR, reason="gateway_error")

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
        return SubmissionAck(status=SubmissionStatus.ERROR, reason="gateway_error")


class CapturingGateway(DryRunGateway):
    """DryRunGateway that captures execute_plan kwargs."""

    def __init__(self) -> None:
        self.calls: list[PlanCall] = []

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        self.calls.append(PlanCall(plan=plan, tick_id=tick_id))
        return await super().execute_plan(plan, tick_id=tick_id)

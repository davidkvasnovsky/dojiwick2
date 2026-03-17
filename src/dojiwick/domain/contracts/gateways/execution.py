"""Execution gateway protocol."""

from typing import Protocol

from dojiwick.domain.enums import OrderTimeInForce, OrderType, PositionSide, TradeAction, WorkingType
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck
from dojiwick.domain.numerics import Price, Quantity


class ExecutionGatewayPort(Protocol):
    """Executes plans and individual orders."""

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        """Execute an execution plan (leg deltas) and return receipts."""
        ...

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        """Cancel an existing order on the exchange."""
        ...

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
        ...

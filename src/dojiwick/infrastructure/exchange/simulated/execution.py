"""Simulated execution gateway for backtest parity with live planner path."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.gateways.clock import ClockPort
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
from dojiwick.domain.models.value_objects.cost_model import CostModel
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck
from dojiwick.domain.numerics import Price, Quantity


@dataclass(slots=True)
class SimulatedExecutionGateway:
    """Implements ExecutionGatewayPort using a CostModel for deterministic fills.

    All orders are immediately filled at slippage-adjusted prices.
    """

    cost_model: CostModel
    prices: dict[str, Decimal]
    clock: ClockPort

    def _slipped_price(self, price: Decimal, side: OrderSide) -> Decimal:
        """Apply slippage to a fill price: adverse for the taker."""
        slip = Decimal(str(self.cost_model.slippage_bps)) / Decimal("10000")
        if side is OrderSide.BUY:
            return price * (1 + slip)
        return price * (1 - slip)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        *,
        tick_id: str = "",
    ) -> tuple[ExecutionReceipt, ...]:
        """Execute all leg deltas with simulated fills."""
        now = self.clock.now_utc()
        receipts: list[ExecutionReceipt] = []
        sorted_deltas = sorted(plan.deltas, key=lambda d: d.sequence)
        for delta in sorted_deltas:
            symbol = delta.instrument_id.symbol
            base_price = self.prices.get(symbol)
            if base_price is None:
                receipts.append(
                    ExecutionReceipt(
                        status=ExecutionStatus.REJECTED,
                        reason=f"no price for {symbol}",
                    )
                )
                continue
            fill_price = self._slipped_price(base_price, delta.side)
            receipts.append(
                ExecutionReceipt(
                    status=ExecutionStatus.FILLED,
                    reason="simulated",
                    fill_price=fill_price,
                    filled_quantity=delta.quantity,
                    order_id=f"sim_{tick_id}_{delta.sequence}",
                    exchange_timestamp=now,
                )
            )
        return tuple(receipts)

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        """No-op cancel in simulation."""
        return SubmissionAck(status=SubmissionStatus.ACCEPTED, order_id=order_id)

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
        """Place a single simulated order."""
        return SubmissionAck(
            status=SubmissionStatus.ACCEPTED,
            order_id=client_order_id or "sim_single",
            exchange_timestamp=self.clock.now_utc(),
        )

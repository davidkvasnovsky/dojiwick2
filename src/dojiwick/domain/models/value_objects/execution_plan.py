"""Execution plan value objects produced by the ExecutionPlannerPort."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.enums import OrderSide, OrderTimeInForce, OrderType, PositionSide, WorkingType
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class LegDelta:
    """A single leg adjustment — the difference between current and target position."""

    instrument_id: InstrumentId
    target_index: int
    position_side: PositionSide
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Price | None = None
    reduce_only: bool = False
    close_position: bool = False
    sequence: int = 0
    time_in_force: OrderTimeInForce = OrderTimeInForce.GTC
    working_type: WorkingType = WorkingType.CONTRACT_PRICE


@dataclass(slots=True, frozen=True, kw_only=True)
class ExecutionPlan:
    """A set of leg deltas that collectively move from current to target positions."""

    account: str
    deltas: tuple[LegDelta, ...]
    estimated_notional: Money = Decimal(0)

    @property
    def is_empty(self) -> bool:
        return len(self.deltas) == 0

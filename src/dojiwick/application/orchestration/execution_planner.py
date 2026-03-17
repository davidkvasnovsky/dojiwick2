"""Default execution planner — computes leg deltas from current vs target positions."""

import logging
from dataclasses import dataclass

from dojiwick.domain.enums import OrderSide, OrderType, PositionMode, PositionSide
from dojiwick.domain.models.value_objects.account_state import AccountSnapshot, ExchangePositionLeg
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, TargetLegPosition
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.numerics import ZERO, Money, Quantity

log = logging.getLogger(__name__)


def _find_leg(
    positions: tuple[ExchangePositionLeg, ...],
    instrument_id: InstrumentId,
    position_side: PositionSide,
) -> ExchangePositionLeg | None:
    """Find matching position leg by instrument + side."""
    for leg in positions:
        if leg.instrument_id == instrument_id and leg.position_side == position_side:
            return leg
    return None


def _compute_estimated_notional(deltas: tuple[LegDelta, ...]) -> Money:
    """Sum quantity * price for all deltas (price=0 for market orders without price)."""
    total = ZERO
    for d in deltas:
        if d.price is not None:
            total += d.quantity * d.price
    return total


@dataclass(slots=True, frozen=True, kw_only=True)
class DefaultExecutionPlanner:
    """Computes leg deltas to move from current positions to target positions.

    Handles both one-way mode (NET-only) and hedge mode (independent LONG/SHORT legs).
    """

    position_mode: PositionMode = PositionMode.ONE_WAY

    async def plan(
        self,
        account_snapshot: AccountSnapshot,
        targets: tuple[TargetLegPosition, ...],
    ) -> ExecutionPlan:
        """Compute the execution plan to move from current positions to targets."""
        deltas: list[LegDelta] = []

        for target_index, target in enumerate(targets):
            target_qty = target.target_qty if target.target_qty is not None else ZERO
            leg_deltas = self._plan_leg(account_snapshot, target, target_qty, target_index)
            deltas.extend(leg_deltas)

        result = tuple(deltas)
        return ExecutionPlan(
            account=account_snapshot.account,
            deltas=result,
            estimated_notional=_compute_estimated_notional(result),
        )

    def _plan_leg(
        self,
        snapshot: AccountSnapshot,
        target: TargetLegPosition,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        """Plan deltas for a single target leg position."""
        if self.position_mode == PositionMode.ONE_WAY:
            return self._plan_one_way(snapshot, target, target_qty, target_index)
        return self._plan_hedge(snapshot, target, target_qty, target_index)

    def _plan_one_way(
        self,
        snapshot: AccountSnapshot,
        target: TargetLegPosition,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        """One-way mode: all positions use NET side."""
        current = _find_leg(snapshot.positions, target.instrument_id, PositionSide.NET)
        current_qty = current.quantity if current is not None else ZERO

        # In one-way mode, LONG targets map to NET with same signed qty
        # NET targets already have signed qty from target_resolver
        return self._compute_net_delta(target.instrument_id, current_qty, target_qty, target_index)

    def _plan_hedge(
        self,
        snapshot: AccountSnapshot,
        target: TargetLegPosition,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        """Hedge mode: independent LONG and SHORT legs."""
        current = _find_leg(snapshot.positions, target.instrument_id, target.position_side)
        current_qty = current.quantity if current is not None else ZERO

        diff = target_qty - current_qty
        if diff == ZERO:
            return []

        if target.position_side == PositionSide.LONG:
            return self._hedge_long_delta(target.instrument_id, diff, target_qty, target_index)
        if target.position_side == PositionSide.SHORT:
            return self._hedge_short_delta(target.instrument_id, diff, target_qty, target_index)

        # NET in hedge mode — treat as one-way
        return self._compute_net_delta(target.instrument_id, current_qty, target_qty, target_index)

    def _hedge_long_delta(
        self,
        instrument_id: InstrumentId,
        diff: Quantity,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        if diff > ZERO:
            return [
                LegDelta(
                    instrument_id=instrument_id,
                    target_index=target_index,
                    position_side=PositionSide.LONG,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=diff,
                )
            ]
        return [
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.LONG,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(diff),
                reduce_only=True,
                close_position=target_qty == ZERO,
            )
        ]

    def _hedge_short_delta(
        self,
        instrument_id: InstrumentId,
        diff: Quantity,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        if diff > ZERO:
            return [
                LegDelta(
                    instrument_id=instrument_id,
                    target_index=target_index,
                    position_side=PositionSide.SHORT,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=diff,
                )
            ]
        return [
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.SHORT,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=abs(diff),
                reduce_only=True,
                close_position=target_qty == ZERO,
            )
        ]

    def _compute_net_delta(
        self,
        instrument_id: InstrumentId,
        current_qty: Quantity,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        """Compute delta for a NET-mode position.

        Signed convention: positive = long, negative = short.
        Direction-aware: correctly handles short-side opens/reduces.
        """
        diff = target_qty - current_qty
        if diff == ZERO:
            return []

        # Flips require two legs with sequencing
        if current_qty > ZERO and target_qty < ZERO:
            return self._plan_flip_long_to_short(instrument_id, current_qty, target_qty, target_index)
        if current_qty < ZERO and target_qty > ZERO:
            return self._plan_flip_short_to_long(instrument_id, current_qty, target_qty, target_index)

        if diff > ZERO:
            if current_qty < ZERO:
                # Reducing short: BUY with reduce_only
                return [
                    LegDelta(
                        instrument_id=instrument_id,
                        target_index=target_index,
                        position_side=PositionSide.NET,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=diff,
                        reduce_only=True,
                        close_position=target_qty == ZERO,
                    )
                ]
            # Increasing long or opening long from flat
            return [
                LegDelta(
                    instrument_id=instrument_id,
                    target_index=target_index,
                    position_side=PositionSide.NET,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=diff,
                )
            ]

        # diff < ZERO
        if current_qty > ZERO:
            # Reducing long: SELL with reduce_only
            return [
                LegDelta(
                    instrument_id=instrument_id,
                    target_index=target_index,
                    position_side=PositionSide.NET,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=abs(diff),
                    reduce_only=True,
                    close_position=target_qty == ZERO,
                )
            ]
        # Increasing short or opening short from flat
        return [
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.NET,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(diff),
            )
        ]

    def _plan_flip_long_to_short(
        self,
        instrument_id: InstrumentId,
        current_qty: Quantity,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        return [
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.NET,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=current_qty,
                reduce_only=True,
                close_position=True,
                sequence=0,
            ),
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.NET,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(target_qty),
                sequence=1,
            ),
        ]

    def _plan_flip_short_to_long(
        self,
        instrument_id: InstrumentId,
        current_qty: Quantity,
        target_qty: Quantity,
        target_index: int,
    ) -> list[LegDelta]:
        return [
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.NET,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=abs(current_qty),
                reduce_only=True,
                close_position=True,
                sequence=0,
            ),
            LegDelta(
                instrument_id=instrument_id,
                target_index=target_index,
                position_side=PositionSide.NET,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=target_qty,
                sequence=1,
            ),
        ]

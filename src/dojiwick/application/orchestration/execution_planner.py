"""Default execution planner — computes leg deltas from current vs target positions."""

import logging
from dataclasses import dataclass, field

from dojiwick.domain.contracts.gateways.exchange_metadata import ExchangeMetadataPort
from dojiwick.domain.enums import OrderSide, OrderType, PositionMode, PositionSide
from dojiwick.domain.models.value_objects.account_state import AccountSnapshot, ExchangePositionLeg
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, TargetLegPosition
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter
from dojiwick.domain.numerics import ZERO, Money, Quantity, meets_min_notional, quantize_qty_to_step

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


@dataclass(slots=True, kw_only=True)
class DefaultExecutionPlanner:
    """Computes leg deltas to move from current positions to target positions.

    Handles both one-way mode (NET-only) and hedge mode (independent LONG/SHORT legs).
    With exchange metadata available, quantities are quantized to the venue's
    step size and sub-minimum entries are dropped — raw sizing output almost
    never conforms to LOT_SIZE and would be rejected order by order.
    """

    position_mode: PositionMode = PositionMode.ONE_WAY
    exchange_metadata: ExchangeMetadataPort | None = None
    _filter_cache: dict[str, InstrumentFilter] = field(default_factory=dict)

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
            for delta in leg_deltas:
                quantized = await self._quantize(delta)
                if quantized is not None:
                    deltas.append(quantized)

        result = tuple(deltas)
        return ExecutionPlan(
            account=account_snapshot.account,
            deltas=result,
            estimated_notional=_compute_estimated_notional(result),
        )

    async def _quantize(self, delta: LegDelta) -> LegDelta | None:
        """Snap quantity to the exchange step grid; drop sub-minimum entries.

        Reduce/close deltas are quantized but never dropped for min-notional —
        exchange minimums do not apply to reduce-only exits, and dropping one
        would strand a position.
        """
        if self.exchange_metadata is None:
            return delta
        filters = await self._filters_for(delta.instrument_id)
        qty = quantize_qty_to_step(delta.quantity, filters.step_size)
        is_reduce = delta.reduce_only or delta.close_position

        if qty <= 0:
            if is_reduce:
                # Residual below one step cannot be closed tighter than the grid
                log.warning("reduce delta for %s below step size — dropped", delta.instrument_id.symbol)
            return None
        if not is_reduce:
            if qty < filters.min_qty:
                log.info("entry for %s below min_qty — dropped", delta.instrument_id.symbol)
                return None
            if filters.max_qty is not None and qty > filters.max_qty:
                qty = quantize_qty_to_step(filters.max_qty, filters.step_size)
            if delta.price is not None and not meets_min_notional(qty, delta.price, filters.min_notional):
                log.info("entry for %s below min notional — dropped", delta.instrument_id.symbol)
                return None
        if qty == delta.quantity:
            return delta
        return LegDelta(
            instrument_id=delta.instrument_id,
            target_index=delta.target_index,
            position_side=delta.position_side,
            side=delta.side,
            order_type=delta.order_type,
            quantity=qty,
            price=delta.price,
            reduce_only=delta.reduce_only,
            close_position=delta.close_position,
            time_in_force=delta.time_in_force,
            working_type=delta.working_type,
            sequence=delta.sequence,
        )

    async def _filters_for(self, instrument_id: InstrumentId) -> InstrumentFilter:
        assert self.exchange_metadata is not None
        cached = self._filter_cache.get(instrument_id.symbol)
        if cached is None:
            info = await self.exchange_metadata.get_instrument(instrument_id)
            cached = info.filters
            self._filter_cache[instrument_id.symbol] = cached
        return cached

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

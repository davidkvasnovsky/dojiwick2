"""Position tracker service — applies fills to position legs."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.contracts.repositories.position_event import PositionEventRepositoryPort
from dojiwick.domain.contracts.repositories.position_leg import PositionLegRepositoryPort
from dojiwick.domain.enums import ExecutionStatus, PositionEventType, PositionSide
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord, PositionLeg
from dojiwick.domain.numerics import Price, Quantity

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionTracker:
    """Applies execution fills to position legs and records lifecycle events."""

    instrument_repo: InstrumentRepositoryPort
    position_leg_repo: PositionLegRepositoryPort
    position_event_repo: PositionEventRepositoryPort
    order_request_repo: OrderRequestRepositoryPort
    clock: ClockPort

    async def apply_fills(
        self,
        plan: ExecutionPlan,
        plan_receipts: tuple[ExecutionReceipt, ...],
        request_ids: dict[int, int],
    ) -> None:
        """Apply filled receipts to position legs via the high-water mark.

        *request_ids* maps delta index -> pre-persisted order_request id.
        """
        resolved_ids: dict[tuple[str, str, str], int | None] = {}

        for i, delta in enumerate(plan.deltas):
            receipt = plan_receipts[i] if i < len(plan_receipts) else None
            if receipt is None or receipt.status is not ExecutionStatus.FILLED:
                continue
            assert receipt.fill_price is not None
            request_id = request_ids.get(i)
            if request_id is None:
                raise AdapterError(f"no pre-persisted order request for filled delta {i}")

            iid = delta.instrument_id
            cache_key = (iid.venue, iid.product, iid.symbol)
            if cache_key not in resolved_ids:
                resolved_ids[cache_key] = await self.instrument_repo.resolve_id(iid.venue, iid.product, iid.symbol)
            instrument_id_int = resolved_ids[cache_key]
            if instrument_id_int is None:
                # This is OUR order on an instrument the DB doesn't know —
                # skipping would silently drop a real fill from position state
                raise AdapterError(f"unknown instrument {iid.venue}/{iid.product}/{iid.symbol} for filled order")

            is_decreasing = delta.reduce_only or delta.close_position
            await self.apply_order_fill(
                order_request_id=request_id,
                cumulative_filled_qty=receipt.filled_quantity,
                fill_price=receipt.fill_price,
                account=plan.account,
                instrument_id=instrument_id_int,
                position_side=delta.position_side,
                is_decreasing=is_decreasing,
            )

    async def apply_order_fill(
        self,
        *,
        order_request_id: int,
        cumulative_filled_qty: Quantity,
        fill_price: Price,
        account: str,
        instrument_id: int,
        position_side: PositionSide,
        is_decreasing: bool,
    ) -> Quantity:
        """Apply an order's cumulative fill through the high-water mark.

        Both the tick path (REST receipt, cumulative executedQty) and the WS
        consumer (cumulative ``z``) call this; whichever arrives second gets a
        zero delta, so fills are never double-applied and partial fills
        compose monotonically. Returns the newly applied quantity.
        """
        delta = await self.order_request_repo.advance_applied_qty(order_request_id, cumulative_filled_qty)
        if delta <= 0:
            return delta
        await self._apply_single_fill(account, instrument_id, position_side, delta, fill_price, is_decreasing)
        return delta

    async def _apply_single_fill(
        self,
        account: str,
        instrument_id: int,
        position_side: PositionSide,
        fill_qty: Quantity,
        fill_price: Price,
        is_decreasing: bool,
    ) -> None:
        """Open, close, reduce, or add to a position leg for one fill."""
        active_leg = await self.position_leg_repo.get_active_leg(account, instrument_id, position_side)
        now = self.clock.now_utc()

        if active_leg is None:
            new_leg = PositionLeg(
                account=account,
                instrument_id=instrument_id,
                position_side=position_side,
                quantity=fill_qty,
                entry_price=fill_price,
                opened_at=now,
            )
            leg_id = await self.position_leg_repo.insert_leg(new_leg)
            await self.position_event_repo.record_event(
                PositionEventRecord(
                    position_leg_id=leg_id,
                    event_type=PositionEventType.OPEN,
                    quantity=fill_qty,
                    price=fill_price,
                    occurred_at=now,
                )
            )
        elif is_decreasing:
            assert active_leg.id is not None
            remaining = active_leg.quantity - fill_qty
            if remaining < 0:
                log.error(
                    "overfill on leg %s: fill %s exceeds quantity %s — closing leg, excess unaccounted",
                    active_leg.id,
                    fill_qty,
                    active_leg.quantity,
                )
            pnl = _compute_pnl(active_leg.position_side, active_leg.entry_price, fill_price, fill_qty)
            if remaining <= 0:
                await self.position_leg_repo.close_leg(active_leg.id, now)
                await self.position_event_repo.record_event(
                    PositionEventRecord(
                        position_leg_id=active_leg.id,
                        event_type=PositionEventType.CLOSE,
                        quantity=fill_qty,
                        price=fill_price,
                        realized_pnl=pnl,
                        occurred_at=now,
                    )
                )
            else:
                await self.position_leg_repo.update_leg(active_leg.id, remaining, active_leg.entry_price)
                await self.position_event_repo.record_event(
                    PositionEventRecord(
                        position_leg_id=active_leg.id,
                        event_type=PositionEventType.REDUCE,
                        quantity=fill_qty,
                        price=fill_price,
                        realized_pnl=pnl,
                        occurred_at=now,
                    )
                )
        else:
            assert active_leg.id is not None
            new_qty = active_leg.quantity + fill_qty
            new_entry = (active_leg.entry_price * active_leg.quantity + fill_price * fill_qty) / new_qty
            await self.position_leg_repo.update_leg(active_leg.id, new_qty, new_entry)
            await self.position_event_repo.record_event(
                PositionEventRecord(
                    position_leg_id=active_leg.id,
                    event_type=PositionEventType.ADD,
                    quantity=fill_qty,
                    price=fill_price,
                    occurred_at=now,
                )
            )


def _compute_pnl(position_side: PositionSide, entry_price: Decimal, fill_price: Decimal, qty: Decimal) -> Decimal:
    """Compute realized PnL for a fill against an entry price.

    NET legs carry signed quantity (negative = short), so the long formula
    with the signed qty yields the correct sign for both directions.
    """
    if position_side is PositionSide.SHORT:
        return (entry_price - fill_price) * abs(qty)
    return (fill_price - entry_price) * qty

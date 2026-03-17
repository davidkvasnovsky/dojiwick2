"""Position tracker service — applies fills to position legs."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.position_event import PositionEventRepositoryPort
from dojiwick.domain.contracts.repositories.position_leg import PositionLegRepositoryPort
from dojiwick.domain.enums import ExecutionStatus, PositionEventType, PositionSide
from dojiwick.domain.numerics import Price, Quantity
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord, PositionLeg

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PositionTracker:
    """Applies execution fills to position legs and records lifecycle events."""

    instrument_repo: InstrumentRepositoryPort
    position_leg_repo: PositionLegRepositoryPort
    position_event_repo: PositionEventRepositoryPort
    clock: ClockPort

    async def apply_fills(
        self,
        plan: ExecutionPlan,
        plan_receipts: tuple[ExecutionReceipt, ...],
    ) -> None:
        """Update position legs based on filled receipts."""
        resolved_ids: dict[tuple[str, str, str], int | None] = {}

        for i, delta in enumerate(plan.deltas):
            receipt = plan_receipts[i] if i < len(plan_receipts) else None
            if receipt is None or receipt.status is not ExecutionStatus.FILLED:
                continue
            assert receipt.fill_price is not None

            iid = delta.instrument_id
            cache_key = (iid.venue, iid.product, iid.symbol)
            if cache_key not in resolved_ids:
                resolved_ids[cache_key] = await self.instrument_repo.resolve_id(iid.venue, iid.product, iid.symbol)
            instrument_id_int = resolved_ids[cache_key]
            if instrument_id_int is None:
                log.warning(
                    "unknown instrument %s/%s/%s — skipping position update", iid.venue, iid.product, iid.symbol
                )
                continue

            is_decreasing = delta.reduce_only or delta.close_position
            await self._apply_single_fill(
                plan.account,
                instrument_id_int,
                delta.position_side,
                receipt.filled_quantity,
                receipt.fill_price,
                is_decreasing,
            )

    async def update_from_fill(
        self,
        account: str,
        instrument_id: int,
        position_side: PositionSide,
        fill_qty: Quantity,
        fill_price: Price,
        is_decreasing: bool,
    ) -> None:
        """Update position legs from a single fill (used by WS event consumer)."""
        await self._apply_single_fill(account, instrument_id, position_side, fill_qty, fill_price, is_decreasing)

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
    """Compute realized PnL for a fill against an entry price."""
    if position_side is PositionSide.LONG or position_side is PositionSide.NET:
        return (fill_price - entry_price) * qty
    return (entry_price - fill_price) * qty

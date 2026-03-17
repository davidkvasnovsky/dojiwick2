"""Exchange-backed reconciliation adapter.

Compares local position state (from DB or in-memory repository) against
the exchange account state (fetched via REST) and returns divergences.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.position_leg import PositionLegRepositoryPort
from dojiwick.domain.enums import PositionSide
from dojiwick.domain.contracts.gateways.account_state import AccountStatePort
from dojiwick.domain.models.value_objects.reconciliation import (
    PositionMismatch,
    ReconciliationResult,
)
from dojiwick.domain.symbols import pair_to_symbol

log = logging.getLogger(__name__)
QUANTITY_EPSILON = Decimal("0.00000001")


@dataclass(slots=True)
class ExchangeReconciliation:
    """Reconciles local DB positions against exchange account state via REST.

    Used at startup to detect divergences before the first tick executes.
    """

    account_state: AccountStatePort
    position_leg_repository: PositionLegRepositoryPort
    instrument_repository: InstrumentRepositoryPort
    account: str
    pair_separator: str = "/"

    async def reconcile(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        """Compare DB positions vs exchange positions for the given pairs."""
        snapshot = await self.account_state.get_account_snapshot(self.account)
        target_symbols = {pair_to_symbol(pair, self.pair_separator) for pair in pairs}

        exchange_positions: dict[tuple[str, PositionSide], Decimal] = {}
        for leg in snapshot.positions:
            symbol = leg.instrument_id.symbol
            if symbol not in target_symbols:
                continue
            key = (symbol, leg.position_side)
            exchange_positions[key] = exchange_positions.get(key, Decimal(0)) + leg.quantity

        db_positions: dict[tuple[str, PositionSide], Decimal] = {}
        all_db_legs = await self.position_leg_repository.get_active_legs(self.account)
        for leg in all_db_legs:
            instrument = await self.instrument_repository.get_by_id(leg.instrument_id)
            if instrument is None:
                log.warning("instrument id not found during reconciliation: %s", leg.instrument_id)
                continue
            symbol = instrument.instrument_id.symbol
            if symbol not in target_symbols:
                continue
            key = (symbol, leg.position_side)
            db_positions[key] = db_positions.get(key, Decimal(0)) + leg.quantity

        all_legs = set(db_positions) | set(exchange_positions)

        orphaned_db: list[str] = []
        orphaned_exchange: list[str] = []
        mismatches: list[PositionMismatch] = []

        for symbol, side in sorted(all_legs, key=lambda value: (value[0], value[1].value)):
            key = (symbol, side)
            in_db = key in db_positions
            on_exchange = key in exchange_positions
            leg_key = f"{symbol}:{side.value}"

            if in_db and not on_exchange:
                orphaned_db.append(leg_key)
                log.warning("orphaned DB position leg: %s", leg_key)
                continue

            if on_exchange and not in_db:
                orphaned_exchange.append(leg_key)
                log.warning("orphaned exchange position leg: %s", leg_key)
                continue

            db_qty = db_positions[key]
            ex_qty = exchange_positions[key]
            if abs(db_qty - ex_qty) <= QUANTITY_EPSILON:
                continue

            mismatches.append(
                PositionMismatch(
                    pair=symbol,
                    order_id="",
                    db_state=f"side={side.value} qty={db_qty}",
                    exchange_state=f"side={side.value} qty={ex_qty}",
                    detail=f"quantity mismatch on {side.value}: db={db_qty} exchange={ex_qty}",
                )
            )
            log.warning("position leg mismatch on %s side=%s: db=%s exchange=%s", symbol, side.value, db_qty, ex_qty)

        result = ReconciliationResult(
            orphaned_db=tuple(orphaned_db),
            orphaned_exchange=tuple(orphaned_exchange),
            mismatches=tuple(mismatches),
        )

        if result.is_clean:
            log.info("reconciliation clean for %d pairs", len(pairs))
        else:
            log.warning(
                "reconciliation divergence: orphaned_db=%d orphaned_exchange=%d mismatches=%d",
                len(orphaned_db),
                len(orphaned_exchange),
                len(mismatches),
            )

        return result

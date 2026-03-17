"""Reconciliation domain models."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class PositionMismatch:
    """A single position state divergence between DB and exchange."""

    pair: str
    order_id: str
    db_state: str
    exchange_state: str
    detail: str = ""


@dataclass(slots=True, frozen=True, kw_only=True)
class ReconciliationResult:
    """Outcome of a reconciliation check."""

    orphaned_db: tuple[str, ...]
    orphaned_exchange: tuple[str, ...]
    mismatches: tuple[PositionMismatch, ...]
    resolved: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.orphaned_db and not self.orphaned_exchange and not self.mismatches

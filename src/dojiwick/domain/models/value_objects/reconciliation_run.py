"""Reconciliation run record for the reconciliation_runs table."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class ReconciliationRun:
    """Result of a reconciliation check stored in reconciliation_runs table."""

    run_type: str
    status: str = "completed"
    orphaned_db_count: int = 0
    orphaned_exchange_count: int = 0
    mismatch_count: int = 0
    resolved_count: int = 0
    divergences: dict[str, object] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.run_type not in ("startup", "periodic"):
            raise ValueError(f"invalid run_type: {self.run_type}")

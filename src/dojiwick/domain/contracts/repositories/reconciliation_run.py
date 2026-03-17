"""Reconciliation run repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.reconciliation_run import ReconciliationRun


class ReconciliationRunRepositoryPort(Protocol):
    """Reconciliation run result persistence."""

    async def insert_run(self, run: ReconciliationRun) -> int:
        """Persist a reconciliation run and return the DB-assigned id."""
        ...

    async def get_latest(self, run_type: str | None = None) -> ReconciliationRun | None:
        """Return the most recent reconciliation run, optionally filtered by type."""
        ...

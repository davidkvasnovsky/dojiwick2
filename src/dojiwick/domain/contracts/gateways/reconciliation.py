"""Reconciliation port for comparing DB state vs exchange state."""

from typing import Protocol

from dojiwick.domain.models.value_objects.reconciliation import ReconciliationResult


class ReconciliationPort(Protocol):
    """Compares local position state against exchange state."""

    async def reconcile(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        """Return divergences between DB positions and exchange positions."""
        ...

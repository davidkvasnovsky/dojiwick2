"""Reconciliation test doubles."""

from dojiwick.domain.models.value_objects.reconciliation import PositionMismatch, ReconciliationResult


class CleanReconciliation:
    """No divergences found."""

    async def reconcile(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        del pairs
        return ReconciliationResult(
            orphaned_db=(),
            orphaned_exchange=(),
            mismatches=(),
        )


class DivergentReconciliation:
    """Always reports divergences."""

    async def reconcile(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        del pairs
        return ReconciliationResult(
            orphaned_db=("BTC/USDC",),
            orphaned_exchange=("ETH/USDC",),
            mismatches=(
                PositionMismatch(
                    pair="SOL/USDC",
                    order_id="order-1",
                    db_state="open",
                    exchange_state="closed",
                    detail="position closed on exchange but open in DB",
                ),
            ),
        )

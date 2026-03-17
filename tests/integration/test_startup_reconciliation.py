"""Integration tests for startup reconciliation gate."""

import pytest

from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.domain.contracts.gateways.reconciliation import ReconciliationPort
from dojiwick.domain.errors import ReconciliationError
from dojiwick.domain.models.value_objects.reconciliation import ReconciliationResult
from fixtures.fakes.audit_log import CapturingAuditLog
from fixtures.fakes.notification import CapturingNotification
from fixtures.fakes.reconciliation import CleanReconciliation, DivergentReconciliation


# Startup gate blocks until reconciliation completes


async def test_startup_gate_completes_when_clean() -> None:
    """Startup gate returns clean result and allows tick loop to proceed."""
    audit = CapturingAuditLog()
    service = ReconciliationService(
        reconciliation_port=CleanReconciliation(),
        audit_log=audit,
    )

    result = await service.run_startup_gate(("BTCUSDC", "ETHUSDC"))

    assert result.is_clean
    assert len(audit.events) == 1
    assert audit.events[0]["event_type"] == "reconciliation_startup"
    ctx = audit.events[0]["context"]
    assert isinstance(ctx, dict)
    assert ctx["is_clean"] is True


async def test_startup_gate_detects_divergences() -> None:
    """Startup gate detects and logs divergences before first tick."""
    audit = CapturingAuditLog()
    notification = CapturingNotification()
    service = ReconciliationService(
        reconciliation_port=DivergentReconciliation(),
        notification=notification,
        audit_log=audit,
    )

    result = await service.run_startup_gate(("BTCUSDC", "ETHUSDC"))

    assert not result.is_clean
    assert len(result.orphaned_db) == 1
    assert len(result.orphaned_exchange) == 1
    assert len(result.mismatches) == 1

    assert len(audit.events) == 1
    ctx = audit.events[0]["context"]
    assert isinstance(ctx, dict)
    assert ctx["is_clean"] is False

    assert len(notification.alerts) == 1
    assert "divergence" in notification.alerts[0][1].lower()


async def test_startup_gate_raises_on_reconciliation_failure() -> None:
    """Startup gate raises ReconciliationError if reconciliation itself fails."""

    class FailingReconciliation(ReconciliationPort):
        async def reconcile(self, pairs: tuple[str, ...]) -> ReconciliationResult:
            del pairs
            raise ConnectionError("exchange unreachable")

    service = ReconciliationService(
        reconciliation_port=FailingReconciliation(),
    )

    with pytest.raises(ReconciliationError, match="startup reconciliation failed"):
        await service.run_startup_gate(("BTCUSDC",))


async def test_startup_gate_persists_result_for_observability() -> None:
    """Reconciliation results are persisted via audit log for observability."""
    audit = CapturingAuditLog()
    service = ReconciliationService(
        reconciliation_port=DivergentReconciliation(),
        audit_log=audit,
    )

    await service.run_startup_gate(("BTCUSDC",))

    assert len(audit.events) == 1
    ctx = audit.events[0]["context"]
    assert isinstance(ctx, dict)
    assert ctx["check_type"] == "startup"
    assert ctx["orphaned_db_count"] == 1
    assert ctx["orphaned_exchange_count"] == 1
    assert ctx["mismatch_count"] == 1


# Backward compatibility: run_startup_check delegates to startup gate


async def test_run_startup_check_delegates_to_gate() -> None:
    """run_startup_check is backward-compatible with run_startup_gate."""
    service = ReconciliationService(
        reconciliation_port=CleanReconciliation(),
    )

    result = await service.run_startup_check(("BTCUSDC",))
    assert result.is_clean


# Periodic check also persists results


async def test_periodic_check_persists_result() -> None:
    """Periodic check persists result for observability."""
    audit = CapturingAuditLog()
    service = ReconciliationService(
        reconciliation_port=CleanReconciliation(),
        audit_log=audit,
    )

    result = await service.run_periodic_check(("BTCUSDC",))

    assert result.is_clean
    assert len(audit.events) == 1
    assert audit.events[0]["event_type"] == "reconciliation_periodic"

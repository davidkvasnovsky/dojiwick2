"""Reconciliation service — compares DB state vs exchange state.

Provides a startup gate: the first tick does not execute until
reconciliation completes successfully. Divergences are detected,
logged, and persisted for observability.
"""

import logging
from dataclasses import dataclass

from dojiwick.domain.enums import AuditSeverity
from dojiwick.domain.errors import ReconciliationError
from dojiwick.domain.models.value_objects.reconciliation import ReconciliationResult
from dojiwick.domain.contracts.gateways.audit_log import AuditLogPort
from dojiwick.domain.contracts.gateways.notification import NotificationPort
from dojiwick.domain.contracts.gateways.reconciliation import ReconciliationPort

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ReconciliationService:
    """Runs startup and periodic reconciliation.

    The startup gate ensures the first tick does not execute until
    reconciliation completes. Results are persisted via AuditLogPort
    for observability.
    """

    reconciliation_port: ReconciliationPort
    notification: NotificationPort | None = None
    audit_log: AuditLogPort | None = None

    async def run_startup_gate(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        """Run reconciliation and block until complete.

        This is the canonical startup gate — the first tick must not
        execute until this method returns. Divergences are detected
        and logged before the first tick is processed. Results are
        persisted for observability.

        Raises ReconciliationError if reconciliation itself fails
        (network error, etc.), ensuring the tick loop does not start.
        """
        log.info("startup reconciliation gate: checking %d pairs", len(pairs))

        try:
            result = await self.reconciliation_port.reconcile(pairs)
        except Exception as exc:
            log.critical("startup reconciliation failed: %s", exc)
            raise ReconciliationError(f"startup reconciliation failed: {exc}") from exc

        await self._persist_result(result, "startup")

        if result.is_clean:
            log.info("startup reconciliation clean — tick loop may proceed")
            return result

        log.warning(
            "startup reconciliation divergence orphaned_db=%d orphaned_exchange=%d mismatches=%d",
            len(result.orphaned_db),
            len(result.orphaned_exchange),
            len(result.mismatches),
        )
        if self.notification is not None:
            await self.notification.send_alert(
                AuditSeverity.WARNING,
                "Reconciliation divergence detected at startup",
                context={
                    "orphaned_db": str(len(result.orphaned_db)),
                    "orphaned_exchange": str(len(result.orphaned_exchange)),
                    "mismatches": str(len(result.mismatches)),
                },
            )
        return result

    async def run_startup_check(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        """Run reconciliation at startup and alert on divergences."""
        return await self.run_startup_gate(pairs)

    async def run_periodic_check(self, pairs: tuple[str, ...]) -> ReconciliationResult:
        """Run periodic reconciliation between ticks."""
        result = await self.reconciliation_port.reconcile(pairs)
        await self._persist_result(result, "periodic")

        if not result.is_clean:
            log.warning(
                "periodic reconciliation divergence orphaned_db=%d orphaned_exchange=%d mismatches=%d",
                len(result.orphaned_db),
                len(result.orphaned_exchange),
                len(result.mismatches),
            )
            if self.notification is not None:
                await self.notification.send_alert(
                    AuditSeverity.WARNING,
                    "Reconciliation divergence detected (periodic)",
                    context={
                        "orphaned_db": str(len(result.orphaned_db)),
                        "orphaned_exchange": str(len(result.orphaned_exchange)),
                        "mismatches": str(len(result.mismatches)),
                    },
                )
        return result

    async def _persist_result(self, result: ReconciliationResult, check_type: str) -> None:
        """Persist reconciliation result for observability via audit log."""
        if self.audit_log is None:
            return

        severity = AuditSeverity.INFO if result.is_clean else AuditSeverity.WARNING
        await self.audit_log.log_event(
            severity=severity,
            event_type=f"reconciliation_{check_type}",
            message=f"Reconciliation {check_type}: {'clean' if result.is_clean else 'divergence detected'}",
            context={
                "check_type": check_type,
                "is_clean": result.is_clean,
                "orphaned_db_count": len(result.orphaned_db),
                "orphaned_exchange_count": len(result.orphaned_exchange),
                "mismatch_count": len(result.mismatches),
                "resolved_count": result.resolved,
                "orphaned_db": list(result.orphaned_db),
                "orphaned_exchange": list(result.orphaned_exchange),
            },
        )

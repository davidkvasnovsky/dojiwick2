"""Startup order cleanup service — cancels stale open orders on restart."""

import logging
from dataclasses import dataclass, replace
from datetime import datetime

from dojiwick.domain.contracts.gateways.audit_log import AuditLogPort
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.open_order import OpenOrderPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.enums import AuditSeverity, OrderEventType, OrderStatus
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.startup_cleanup import StaleOrderRecord, StartupCleanupResult
from dojiwick.domain.order_state_machine import is_terminal

log = logging.getLogger(__name__)


@dataclass(slots=True)
class StartupOrderCleanupService:
    """Cancels stale open orders found at startup and reconciles the order ledger."""

    open_order_port: OpenOrderPort
    order_report_repo: OrderReportRepositoryPort
    order_event_repo: OrderEventRepositoryPort
    audit_log: AuditLogPort
    clock: ClockPort

    async def run(self, symbols: tuple[str, ...]) -> StartupCleanupResult:
        """Cancel stale open orders across all symbols and reconcile DB state."""
        all_cancelled: list[StaleOrderRecord] = []
        all_errors: list[str] = []

        for symbol in symbols:
            try:
                cancelled, errors = await self._cleanup_symbol(symbol)
                all_cancelled.extend(cancelled)
                all_errors.extend(errors)
            except AdapterError as exc:
                msg = f"{symbol}: {exc}"
                log.error("startup cleanup exchange error: %s", msg)
                all_errors.append(msg)

        result = StartupCleanupResult(
            cancelled=tuple(all_cancelled),
            errors=tuple(all_errors),
        )

        severity = AuditSeverity.INFO if result.is_clean else AuditSeverity.WARNING
        await self.audit_log.log_event(
            severity=severity,
            event_type="startup_order_cleanup",
            message=f"cancelled={len(result.cancelled)} errors={len(result.errors)}",
            context={
                "cancelled_count": len(result.cancelled),
                "error_count": len(result.errors),
                "symbols": list(symbols),
            },
        )

        return result

    async def _cleanup_symbol(self, symbol: str) -> tuple[list[StaleOrderRecord], list[str]]:
        """Clean up stale orders for a single symbol."""
        cancelled: list[StaleOrderRecord] = []
        errors: list[str] = []

        open_orders = await self.open_order_port.get_open_orders(symbol)
        if not open_orders:
            return cancelled, errors

        log.warning(
            "startup: found %d open orders for %s (abnormal for market-only engine)",
            len(open_orders),
            symbol,
        )

        await self.open_order_port.cancel_all_open_orders(symbol)

        now = self.clock.now_utc()
        for order in open_orders:
            record = StaleOrderRecord(
                symbol=symbol,
                exchange_order_id=order.exchange_order_id,
                client_order_id=order.client_order_id,
                status=order.status,
                filled_quantity=order.filled_quantity,
            )
            cancelled.append(record)

            try:
                await self._reconcile_db_report(order.exchange_order_id, now)
            except Exception as exc:
                msg = f"{symbol}/{order.exchange_order_id}: reconcile failed: {exc}"
                log.warning("startup cleanup: %s", msg)
                errors.append(msg)

        return cancelled, errors

    async def _reconcile_db_report(self, exchange_order_id: str, now: datetime) -> None:
        """Update DB report to CANCELED if it exists and is non-terminal."""
        existing = await self.order_report_repo.get_by_exchange_order_id(exchange_order_id)
        if existing is None:
            log.warning(
                "startup cleanup: no DB report for exchange_order_id=%s (order placed but response lost)",
                exchange_order_id,
            )
            return

        if is_terminal(existing.status):
            return

        updated = replace(existing, status=OrderStatus.CANCELED, reported_at=now)
        await self.order_report_repo.upsert_report(updated)

        assert existing.id is not None
        event = OrderEvent(
            order_id=existing.order_request_id,
            event_type=OrderEventType.CANCELED,
            occurred_at=now,
            exchange_order_id=exchange_order_id,
            detail="startup_cleanup",
        )
        await self.order_event_repo.record_event(event)

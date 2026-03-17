"""Startup crash-recovery orchestrator.

Extracts the inline startup logic from the runner into a structured
application service. Handles feed bootstrap, order cleanup, reconciliation,
cursor-based event replay, health evaluation, and consumer creation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from dojiwick.application.models.startup_result import StartupResult
from dojiwick.application.services.order_event_consumer import OrderEventConsumer
from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.application.services.startup_order_cleanup import StartupOrderCleanupService
from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.domain.contracts.gateways.audit_log import AuditLogPort
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.market_data_feed import MarketDataFeedPort
from dojiwick.domain.contracts.gateways.open_order import OpenOrderPort
from dojiwick.domain.contracts.gateways.order_event_stream import (
    OrderEventStreamPort,
    StreamCursor,
)
from dojiwick.domain.contracts.repositories.bot_state import BotStateRepositoryPort
from dojiwick.domain.contracts.repositories.fill import FillRepositoryPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.contracts.repositories.stream_cursor import StreamCursorRepositoryPort
from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.models.value_objects.health_state import HealthState
from dojiwick.domain.models.value_objects.reconciliation import ReconciliationResult
from dojiwick.domain.reconciliation_health import compute_health

log = logging.getLogger(__name__)

_BOOTSTRAP_MAX_RETRIES = 3


@dataclass(slots=True)
class StartupOrchestrator:
    """Runs the full crash-recovery startup sequence."""

    feed: MarketDataFeedPort | None
    order_stream: OrderEventStreamPort | None
    open_order_port: OpenOrderPort | None
    reconciliation_service: ReconciliationService
    order_request_repo: OrderRequestRepositoryPort
    order_report_repo: OrderReportRepositoryPort
    fill_repo: FillRepositoryPort
    order_event_repo: OrderEventRepositoryPort
    cursor_repo: StreamCursorRepositoryPort
    bot_state_repository: BotStateRepositoryPort
    position_tracker: PositionTracker
    clock: ClockPort
    audit_log: AuditLogPort

    pair_symbols: tuple[str, ...]
    degraded_timeout_sec: int
    uncertain_timeout_sec: int

    async def run(self) -> StartupResult:
        """Execute the full startup sequence and return the result."""
        cancelled_orders = 0
        replayed_events = 0
        periodic_reconciliation: ReconciliationService | None = None

        if self.feed is None:
            health = HealthState()
            return StartupResult(
                health=health,
                consumer_task=None,
                periodic_reconciliation=None,
                cancelled_orders=0,
                replayed_events=0,
            )

        # 1. Bootstrap feed
        await self._bootstrap_feed()
        await self.feed.start()

        # 2. Cancel stale orders
        cancelled_orders = await self._cancel_stale_orders()

        # 3. Initial reconciliation
        recon_result = await self.reconciliation_service.run_startup_gate(self.pair_symbols)

        # 4. Replay missed stream events
        replayed_events = await self._replay_missed_events()

        # 5. Re-reconcile after replay
        if replayed_events > 0:
            recon_result = await self.reconciliation_service.run_startup_gate(self.pair_symbols)

        # 6. Compute health
        health = await self._compute_health(recon_result)

        # 7. Health gate
        if health.health in (ReconciliationHealth.UNCERTAIN, ReconciliationHealth.HALT):
            return StartupResult(
                health=health,
                consumer_task=None,
                periodic_reconciliation=None,
                cancelled_orders=cancelled_orders,
                replayed_events=replayed_events,
            )

        periodic_reconciliation = self.reconciliation_service

        # 8. Start consumer
        consumer_task = await self._start_consumer()

        return StartupResult(
            health=health,
            consumer_task=consumer_task,
            periodic_reconciliation=periodic_reconciliation,
            cancelled_orders=cancelled_orders,
            replayed_events=replayed_events,
        )

    async def _bootstrap_feed(self) -> None:
        """Bootstrap the feed with retries."""
        assert self.feed is not None
        for attempt in range(1, _BOOTSTRAP_MAX_RETRIES + 1):
            try:
                await self.feed.bootstrap()
                return
            except Exception:
                if attempt == _BOOTSTRAP_MAX_RETRIES:
                    raise
                log.warning("bootstrap attempt %d/%d failed, retrying", attempt, _BOOTSTRAP_MAX_RETRIES, exc_info=True)
                await asyncio.sleep(2**attempt)

    async def _cancel_stale_orders(self) -> int:
        """Cancel stale open orders. Returns the number cancelled."""
        if self.open_order_port is None:
            return 0

        cleanup_service = StartupOrderCleanupService(
            open_order_port=self.open_order_port,
            order_report_repo=self.order_report_repo,
            order_event_repo=self.order_event_repo,
            audit_log=self.audit_log,
            clock=self.clock,
        )
        cleanup_result = await cleanup_service.run(self.pair_symbols)
        if cleanup_result.cancelled:
            log.warning("startup cleanup: cancelled %d stale orders", len(cleanup_result.cancelled))
        if cleanup_result.errors:
            log.error("startup cleanup errors: %s", cleanup_result.errors)
        return len(cleanup_result.cancelled)

    async def _replay_missed_events(self) -> int:
        """Replay missed order events from the last cursor. Returns count."""
        if self.order_stream is None:
            return 0

        cursor_record = await self.cursor_repo.get_cursor(self.order_stream.stream_name)
        if cursor_record is None:
            return 0

        timestamp_ms = 0
        if cursor_record.last_event_time is not None:
            timestamp_ms = int(cursor_record.last_event_time.timestamp() * 1000)

        cursor = StreamCursor(
            stream_name=self.order_stream.stream_name,
            sequence=0,
            timestamp_ms=timestamp_ms,
        )

        events = [e async for e in self.order_stream.replay_from(cursor)]
        if not events:
            return 0

        reports = await self.order_report_repo.get_by_exchange_order_ids(
            [e.exchange_order_id for e in events],
        )

        count = 0
        for event in events:
            report = reports.get(event.exchange_order_id)
            if report is None:
                log.warning("replay: unknown exchange_order_id=%s — skipping", event.exchange_order_id)
                continue
            corrected = replace(event, order_id=report.order_request_id)
            await self.order_event_repo.record_event(corrected)
            count += 1

        if count > 0:
            log.info("replayed %d missed order events", count)

        return count

    async def _compute_health(
        self,
        recon_result: ReconciliationResult,
    ) -> HealthState:
        """Load bot state, compute health, persist, and return."""
        state = await self.bot_state_repository.get_state()
        if recon_result.is_clean:
            normal = HealthState()
            if state.recon_health is not ReconciliationHealth.NORMAL:
                normal.apply_to(state)
                await self.bot_state_repository.update_state(state)
            return normal
        current_hs = HealthState.from_bot_state(state)
        new_hs = compute_health(
            current_hs,
            recon_result,
            self.clock.now_utc(),
            degraded_timeout_sec=self.degraded_timeout_sec,
            uncertain_timeout_sec=self.uncertain_timeout_sec,
        )
        new_hs.apply_to(state)
        await self.bot_state_repository.update_state(state)
        return new_hs

    async def _start_consumer(self) -> asyncio.Task[None] | None:
        """Start the order event consumer background task."""
        if self.order_stream is None:
            return None

        consumer = OrderEventConsumer(
            stream=self.order_stream,
            order_request_repo=self.order_request_repo,
            order_report_repo=self.order_report_repo,
            fill_repo=self.fill_repo,
            order_event_repo=self.order_event_repo,
            position_tracker=self.position_tracker,
            cursor_repo=self.cursor_repo,
            clock=self.clock,
        )
        task = asyncio.create_task(consumer.run())
        log.info("order event consumer started")
        return task

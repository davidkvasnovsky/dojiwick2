"""Long-running batch runner with PostgreSQL persistence."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.services.order_ledger import OrderLedgerService
from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.application.services.startup_orchestrator import StartupOrchestrator
from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.config.composition import ComposedAdapters, build_adapters
from dojiwick.config.schema import Settings
from dojiwick.config.fingerprint import SettingsFingerprint, fingerprint_settings
from dojiwick.config.loader import load_settings
from dojiwick.config.logging import configure_logging
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.market_data_feed import MarketDataFeedPort
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.errors import (
    AdapterError,
    AuthenticationError,
    CircuitBreakerTrippedError,
    DataQualityError,
    ExchangeError,
    InsufficientBalanceError,
    NetworkError,
    PostExecutionPersistenceError,
)
from dojiwick.config.targets import resolve_execution_symbols, resolve_instrument_map, resolve_target_ids
from dojiwick.domain.hashing import compute_tick_id
from dojiwick.domain.models.value_objects.health_state import HealthState
from dojiwick.domain.reconciliation_health import compute_health
from dojiwick.infrastructure.ai.factory import AIServices, build_ai_services
from dojiwick.infrastructure.observability.alert_evaluator import AlertEvaluator
from dojiwick.infrastructure.observability.log_notification import LogNotification
from dojiwick.infrastructure.observability.null_metrics import NullMetrics
from dojiwick.infrastructure.system.clock import SystemClock
from dojiwick.infrastructure.postgres.connection import (
    DbConnection,
    TransactionAwareConnection,
    check_db_connectivity,
    create_pool,
)
from dojiwick.infrastructure.exchange.reconciliation import ExchangeReconciliation
from dojiwick.infrastructure.postgres.gateways.audit_log import PgAuditLog
from dojiwick.infrastructure.postgres.repositories.bot_config_snapshot import PgBotConfigSnapshotRepository
from dojiwick.infrastructure.postgres.repositories.decision_trace import PgDecisionTraceRepository
from dojiwick.infrastructure.postgres.repositories.instrument import PgInstrumentRepository
from dojiwick.infrastructure.postgres.repositories.fill import PgFillRepository
from dojiwick.infrastructure.postgres.repositories.order_event import PgOrderEventRepository
from dojiwick.infrastructure.postgres.repositories.order_report import PgOrderReportRepository
from dojiwick.infrastructure.postgres.repositories.order_request import PgOrderRequestRepository
from dojiwick.infrastructure.postgres.pending_order_provider import PgPendingOrderProvider
from dojiwick.infrastructure.postgres.repositories.outcome import PgOutcomeRepository
from dojiwick.infrastructure.postgres.repositories.position_event import PgPositionEventRepository
from dojiwick.infrastructure.postgres.repositories.position_leg import PgPositionLegRepository
from dojiwick.infrastructure.postgres.repositories.regime import PgRegimeRepository
from dojiwick.infrastructure.postgres.repositories.bot_state import PgBotStateRepository
from dojiwick.infrastructure.postgres.repositories.stream_cursor import PgStreamCursorRepository
from dojiwick.infrastructure.postgres.repositories.tick import PgTickRepository
from dojiwick.infrastructure.postgres.unit_of_work import PgUnitOfWork

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for loop runner."""

    parser = argparse.ArgumentParser(description="Run periodic dojiwick engine ticks with postgres persistence")
    parser.add_argument("--config", default="config.toml")
    return parser


@dataclass(frozen=True, slots=True)
class _WiredServices:
    service: TickService
    ai_services: AIServices
    bot_state_repository: PgBotStateRepository
    tick_repository: PgTickRepository
    orchestrator: StartupOrchestrator
    _cleanup: Callable[[], Awaitable[None]] = field(repr=False)

    async def cleanup(self) -> None:
        await self._cleanup()


async def _wire_services(
    settings: Settings,
    fingerprint: SettingsFingerprint,
    clock: ClockPort,
    adapters: ComposedAdapters,
    pool: AsyncConnectionPool[Any],
    notification: LogNotification,
    pair_symbols: tuple[str, ...],
    target_ids: tuple[str, ...],
    instrument_map: dict[str, InstrumentId] | None = None,  # kept optional for signature compat
) -> _WiredServices:
    """Create all repositories, services, and connections."""
    main_raw = cast(DbConnection, await pool.getconn())
    main_uow = PgUnitOfWork(connection=main_raw)
    main_conn = TransactionAwareConnection(inner=main_raw, unit_of_work=main_uow)

    consumer_raw = cast(DbConnection, await pool.getconn())
    consumer_conn = TransactionAwareConnection(inner=consumer_raw, unit_of_work=None)

    position_leg_repository = PgPositionLegRepository(connection=main_conn)
    instrument_repository = PgInstrumentRepository(connection=main_conn)
    audit_log = PgAuditLog(connection=main_conn)
    config_snapshot_repository = PgBotConfigSnapshotRepository(connection=main_conn)
    await config_snapshot_repository.record_snapshot(fingerprint.sha256, fingerprint.canonical_json)
    log.info(
        "config snapshot recorded full_hash=%s trading_hash=%s",
        fingerprint.sha256,
        fingerprint.trading_sha256,
    )

    order_request_repo = PgOrderRequestRepository(connection=main_conn)
    order_report_repo = PgOrderReportRepository(connection=main_conn)
    fill_repo = PgFillRepository(connection=main_conn)
    order_event_repo = PgOrderEventRepository(connection=main_conn)

    order_ledger = OrderLedgerService(
        instrument_repo=instrument_repository,
        order_request_repo=order_request_repo,
        order_report_repo=order_report_repo,
        fill_repo=fill_repo,
        order_event_repo=order_event_repo,
        clock=clock,
    )

    def _build_position_tracker(conn: TransactionAwareConnection) -> PositionTracker:
        return PositionTracker(
            instrument_repo=PgInstrumentRepository(connection=conn),
            position_leg_repo=PgPositionLegRepository(connection=conn),
            position_event_repo=PgPositionEventRepository(connection=conn),
            clock=clock,
        )

    position_tracker = _build_position_tracker(main_conn)

    consumer_order_request_repo = PgOrderRequestRepository(connection=consumer_conn)
    consumer_order_report_repo = PgOrderReportRepository(connection=consumer_conn)
    consumer_fill_repo = PgFillRepository(connection=consumer_conn)
    consumer_order_event_repo = PgOrderEventRepository(connection=consumer_conn)
    consumer_cursor_repo = PgStreamCursorRepository(connection=consumer_conn)
    consumer_position_tracker = _build_position_tracker(consumer_conn)

    ai_services = build_ai_services(settings.ai, clock=clock)
    bot_state_repository = PgBotStateRepository(connection=main_conn)
    tick_repository = PgTickRepository(connection=main_conn)

    service = TickService(
        settings=settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(settings.risk),
        clock=clock,
        context_provider=adapters.context_provider,
        execution_gateway=adapters.execution_gateway,
        execution_planner=adapters.execution_planner,
        account_state_provider=adapters.account_state,
        outcome_repository=PgOutcomeRepository(connection=main_conn),
        regime_repository=PgRegimeRepository(connection=main_conn),
        tick_repository=tick_repository,
        bot_state_repository=bot_state_repository,
        veto_service=ai_services.veto_service,
        regime_classifier=ai_services.regime_classifier,
        metrics=NullMetrics(),
        unit_of_work=main_uow,
        decision_trace_repository=PgDecisionTraceRepository(connection=main_conn),
        order_ledger=order_ledger,
        position_tracker=position_tracker,
        pending_order_provider=PgPendingOrderProvider(connection=main_conn),
        config_hash=fingerprint.sha256,
        target_ids=target_ids,
        instrument_map=instrument_map,
    )

    reconciliation_service = ReconciliationService(
        reconciliation_port=ExchangeReconciliation(
            account_state=adapters.account_state,
            position_leg_repository=position_leg_repository,
            instrument_repository=instrument_repository,
            account=settings.universe.account,
            pair_separator=settings.universe.pair_separator,
        ),
        notification=notification,
        audit_log=audit_log,
    )

    orchestrator = StartupOrchestrator(
        feed=adapters.feed,
        order_stream=adapters.order_stream,
        open_order_port=adapters.open_order_port,
        reconciliation_service=reconciliation_service,
        order_request_repo=consumer_order_request_repo,
        order_report_repo=consumer_order_report_repo,
        fill_repo=consumer_fill_repo,
        order_event_repo=consumer_order_event_repo,
        cursor_repo=consumer_cursor_repo,
        bot_state_repository=bot_state_repository,
        position_tracker=consumer_position_tracker,
        clock=clock,
        audit_log=audit_log,
        pair_symbols=pair_symbols,
        degraded_timeout_sec=settings.system.recon_degraded_timeout_sec,
        uncertain_timeout_sec=settings.system.recon_uncertain_timeout_sec,
    )

    async def _do_cleanup() -> None:
        await main_uow.rollback_if_active()
        await pool.putconn(main_raw)
        await pool.putconn(consumer_raw)

    return _WiredServices(
        service=service,
        ai_services=ai_services,
        bot_state_repository=bot_state_repository,
        tick_repository=tick_repository,
        orchestrator=orchestrator,
        _cleanup=_do_cleanup,
    )


async def _run_periodic_recon(
    *,
    periodic_recon: ReconciliationService,
    pair_symbols: tuple[str, ...],
    bot_state_repository: PgBotStateRepository,
    clock: ClockPort,
    settings: Settings,
) -> bool:
    """Run periodic reconciliation. Returns True if HALT detected."""
    try:
        recon_result = await periodic_recon.run_periodic_check(pair_symbols)
        state = await bot_state_repository.get_state()
        current_hs = HealthState.from_bot_state(state)
        new_hs = compute_health(
            current_hs,
            recon_result,
            clock.now_utc(),
            degraded_timeout_sec=settings.system.recon_degraded_timeout_sec,
            uncertain_timeout_sec=settings.system.recon_uncertain_timeout_sec,
        )
        if new_hs.health != current_hs.health:
            log.warning(
                "reconciliation health %s -> %s frozen=%s",
                current_hs.health,
                new_hs.health,
                new_hs.frozen_symbols,
            )
        new_hs.apply_to(state)
        await bot_state_repository.update_state(state)
        if new_hs.health is ReconciliationHealth.HALT:
            log.critical("reconciliation HALT — manual intervention required")
            return True
    except (AdapterError, OSError, asyncio.TimeoutError) as exc:
        log.warning("periodic reconciliation failed (retryable): %s", exc)
    except Exception:
        log.exception("unexpected error in reconciliation — crashing")
        raise
    return False


async def _run_tick_loop(
    *,
    service: TickService,
    feed: MarketDataFeedPort | None,
    settings: Settings,
    stop_event: asyncio.Event,
    alert_evaluator: AlertEvaluator,
    ai_services: AIServices,
    clock: ClockPort,
    consumer_task: asyncio.Task[None] | None,
    fingerprint: SettingsFingerprint,
    recon_check: Callable[[], Awaitable[bool]] | None,
) -> int:
    """Execute main tick loop. Returns tick_count."""
    max_consecutive_errors = settings.system.max_consecutive_errors
    consecutive_errors = 0
    tick_count = 0
    max_ticks = settings.system.max_ticks
    recon_interval = settings.system.reconciliation_interval_ticks
    cost_tracker = ai_services.cost_tracker

    while not stop_event.is_set():
        if consumer_task is not None and consumer_task.done():
            if consumer_task.cancelled():
                log.critical("order event consumer was cancelled unexpectedly")
            else:
                exc = consumer_task.exception()
                log.critical("order event consumer died: %s", exc)
            break

        try:
            if feed is not None:
                await feed.ensure_fresh()
            observed_at = clock.now_utc()
            if cost_tracker is not None:
                cost_tracker.current_tick_id = compute_tick_id(
                    fingerprint.sha256, observed_at, settings.trading.active_pairs
                )
            await service.run_tick(at=observed_at)
            consecutive_errors = 0
            tick_count += 1
            if cost_tracker is not None:
                await cost_tracker.flush()
                alert_evaluator.evaluate_budget(
                    cost_tracker.day_spend_usd,
                    settings.ai.daily_budget_usd,
                )
        except PostExecutionPersistenceError:
            log.critical("post-execution persistence failure — halting")
            break
        except (AuthenticationError, InsufficientBalanceError) as exc:
            log.critical("fatal exchange error, halting: %s", exc)
            break
        except (
            DataQualityError,
            CircuitBreakerTrippedError,
            NetworkError,
            ExchangeError,
            AdapterError,
            OSError,
            asyncio.TimeoutError,
        ) as exc:
            consecutive_errors += 1
            alert_evaluator.evaluate_tick_failure(consecutive_errors)
            log.warning("tick failed (retryable) error=%s n=%d", type(exc).__name__, consecutive_errors)
            if consecutive_errors >= max_consecutive_errors:
                log.critical("halting after %d consecutive tick failures", consecutive_errors)
                break
        except Exception:
            log.exception("unexpected error in tick loop — crashing")
            raise

        if tick_count > 0 and tick_count % recon_interval == 0 and recon_check is not None:
            if await recon_check():
                break
        if max_ticks > 0 and tick_count >= max_ticks:
            log.info("max ticks reached ticks=%d", tick_count)
            break

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(settings.system.tick_interval_sec))
        except TimeoutError:
            continue

    return tick_count


async def _run(config_path: Path) -> int:
    settings = load_settings(config_path)
    fingerprint = fingerprint_settings(settings)
    configure_logging(level=settings.system.log_level)

    clock = SystemClock()
    alert_evaluator = AlertEvaluator()
    notification = LogNotification()
    adapters = build_adapters(settings, clock=clock)

    pair_symbols = resolve_execution_symbols(settings)
    target_ids = resolve_target_ids(settings)
    instrument_map = resolve_instrument_map(settings)

    pool = await create_pool(settings.database)
    await check_db_connectivity(pool)
    wired = await _wire_services(
        settings, fingerprint, clock, adapters, pool, notification, pair_symbols, target_ids, instrument_map
    )

    recovered = await wired.tick_repository.recover_stale_started()
    if recovered:
        log.warning("recovered %d stale STARTED ticks at startup", recovered)

    startup_result = await wired.orchestrator.run()

    if startup_result.health.health in (ReconciliationHealth.UNCERTAIN, ReconciliationHealth.HALT):
        log.critical("startup health %s — refusing to start", startup_result.health.health)
        await wired.cleanup()
        await pool.close()
        return 1

    periodic_reconciliation = startup_result.periodic_reconciliation
    consumer_task = startup_result.consumer_task

    if periodic_reconciliation is not None:

        async def _recon_check() -> bool:
            return await _run_periodic_recon(
                periodic_recon=periodic_reconciliation,
                pair_symbols=pair_symbols,
                bot_state_repository=wired.bot_state_repository,
                clock=clock,
                settings=settings,
            )

        recon_check: Callable[[], Awaitable[bool]] | None = _recon_check
    else:
        recon_check = None

    stop_event = asyncio.Event()
    wired.service.shutdown_event = stop_event
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # pragma: no cover
            pass

    try:
        tick_count = await _run_tick_loop(
            service=wired.service,
            feed=adapters.feed,
            settings=settings,
            stop_event=stop_event,
            alert_evaluator=alert_evaluator,
            ai_services=wired.ai_services,
            clock=clock,
            consumer_task=consumer_task,
            fingerprint=fingerprint,
            recon_check=recon_check,
        )
    finally:
        log.info("shutting down gracefully timeout=%ds", settings.system.shutdown_timeout_sec)
        if consumer_task is not None and not consumer_task.done():
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        await adapters.close()
        await wired.cleanup()
        await pool.close()

    log.info("runner stopped ticks=%d", tick_count)
    return 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()  # runner has its own logging via configure_logging()
    args = build_parser().parse_args(argv)
    return asyncio.run(_run(Path(args.config)))


if __name__ == "__main__":
    raise SystemExit(main())

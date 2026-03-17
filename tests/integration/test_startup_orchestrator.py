"""Integration tests for the StartupOrchestrator."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from dojiwick.application.models.startup_result import StartupResult
from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.application.services.startup_orchestrator import StartupOrchestrator
from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.domain.enums import (
    OrderEventType,
    OrderSide,
    OrderStatus,
    PositionSide,
    ReconciliationHealth,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.contracts.gateways.market_data_feed import MarketDataFeedPort
from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder
from dojiwick.domain.models.value_objects.account_state import (
    AccountSnapshot,
    ExchangePositionLeg,
)
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.order_request import OrderReport
from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord
from dojiwick.infrastructure.exchange.reconciliation import ExchangeReconciliation
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.audit_log import CapturingAuditLog
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.exchange_data_feed import FakeExchangeDataFeed
from fixtures.fakes.fill_repository import FakeFillRepo
from fixtures.fakes.instrument_repository import FakeInstrumentRepo
from fixtures.fakes.notification import CapturingNotification
from fixtures.fakes.open_order import FakeOpenOrderAdapter
from fixtures.fakes.order_event_repository import FakeOrderEventRepository
from fixtures.fakes.order_event_stream import InMemoryOrderEventStream
from fixtures.fakes.order_report_repository import FakeOrderReportRepo
from fixtures.fakes.order_request_repository import FakeOrderRequestRepo
from fixtures.fakes.position_event_repository import FakePositionEventRepo
from fixtures.fakes.position_leg_repository import FakePositionLegRepo
from fixtures.fakes.stream_cursor_repository import FakeStreamCursorRepo

PAIR_SYMBOLS = ("BTCUSDC",)


def _empty_snapshot(account: str = "test") -> AccountSnapshot:
    return AccountSnapshot(
        account=account,
        balances=(),
        positions=(),
        total_wallet_balance=Decimal(0),
        available_balance=Decimal(0),
    )


async def _cancel_consumer(result: StartupResult) -> None:
    """Cancel consumer task if present and wait for it to finish."""
    if result.consumer_task is not None:
        result.consumer_task.cancel()
        try:
            await result.consumer_task
        except asyncio.CancelledError:
            pass


def _build_orchestrator(
    *,
    feed: MarketDataFeedPort | None = None,
    order_stream: InMemoryOrderEventStream | None = None,
    open_order_port: FakeOpenOrderAdapter | None = None,
    account_state: FakeAccountState | None = None,
    position_leg_repo: FakePositionLegRepo | None = None,
    instrument_repo: FakeInstrumentRepo | None = None,
    order_request_repo: FakeOrderRequestRepo | None = None,
    order_report_repo: FakeOrderReportRepo | None = None,
    fill_repo: FakeFillRepo | None = None,
    order_event_repo: FakeOrderEventRepository | None = None,
    cursor_repo: FakeStreamCursorRepo | None = None,
    bot_state_repo: InMemoryBotStateRepo | None = None,
    clock: FixedClock | None = None,
    audit_log: CapturingAuditLog | None = None,
    notification: CapturingNotification | None = None,
    reconciliation_service_override: ReconciliationService | None = None,
    pair_symbols: tuple[str, ...] = PAIR_SYMBOLS,
    account: str = "test",
) -> StartupOrchestrator:
    acct = account_state or FakeAccountState()
    if account_state is None:
        acct.set_snapshot(account, _empty_snapshot(account))

    clk = clock or FixedClock()
    inst_repo = instrument_repo or FakeInstrumentRepo()
    pos_leg_repo = position_leg_repo or FakePositionLegRepo()
    aud = audit_log or CapturingAuditLog()
    notif = notification or CapturingNotification()

    position_tracker = PositionTracker(
        instrument_repo=inst_repo,
        position_leg_repo=pos_leg_repo,
        position_event_repo=FakePositionEventRepo(),
        clock=clk,
    )

    recon_svc = reconciliation_service_override or ReconciliationService(
        reconciliation_port=ExchangeReconciliation(
            account_state=acct,
            position_leg_repository=pos_leg_repo,
            instrument_repository=inst_repo,
            account=account,
            pair_separator="/",
        ),
        notification=notif,
        audit_log=aud,
    )

    return StartupOrchestrator(
        feed=feed,
        order_stream=order_stream,
        open_order_port=open_order_port,
        reconciliation_service=recon_svc,
        order_request_repo=order_request_repo or FakeOrderRequestRepo(),
        order_report_repo=order_report_repo or FakeOrderReportRepo(),
        fill_repo=fill_repo or FakeFillRepo(),
        order_event_repo=order_event_repo or FakeOrderEventRepository(),
        cursor_repo=cursor_repo or FakeStreamCursorRepo(),
        bot_state_repository=bot_state_repo or InMemoryBotStateRepo(),
        position_tracker=position_tracker,
        clock=clk,
        audit_log=aud,
        pair_symbols=pair_symbols,
        degraded_timeout_sec=300,
        uncertain_timeout_sec=900,
    )


# 1. Clean startup → NORMAL health


async def test_clean_startup_normal_health() -> None:
    """Clean reconciliation → NORMAL health, consumer task started, periodic set."""
    feed = FakeExchangeDataFeed()
    stream = InMemoryOrderEventStream()
    orch = _build_orchestrator(feed=feed, order_stream=stream)

    result = await orch.run()

    assert result.health.health is ReconciliationHealth.NORMAL
    assert result.periodic_reconciliation is not None
    assert result.consumer_task is not None
    assert result.cancelled_orders == 0
    assert result.replayed_events == 0
    assert feed.bootstrapped
    assert feed.started
    await _cancel_consumer(result)


# 2. No feed → skip reconciliation, NORMAL health


async def test_startup_no_feed_skips_reconciliation() -> None:
    """feed=None → NORMAL health, no reconciliation, no consumer."""
    orch = _build_orchestrator(feed=None, order_stream=None)

    result = await orch.run()

    assert result.health.health is ReconciliationHealth.NORMAL
    assert result.periodic_reconciliation is None
    assert result.consumer_task is None


# 3. Bootstrap retry on failure


async def test_bootstrap_retry_on_failure() -> None:
    """Feed bootstrap fails once then succeeds → orchestrator succeeds."""
    feed = FakeExchangeDataFeed(bootstrap_error=ConnectionError("flaky"))
    orch = _build_orchestrator(feed=feed, order_stream=InMemoryOrderEventStream())

    result = await orch.run()

    assert feed.bootstrapped
    assert result.health.health is ReconciliationHealth.NORMAL


async def test_bootstrap_retry_cleanup() -> None:
    """Clean up consumer task from retry test."""
    feed = FakeExchangeDataFeed(bootstrap_error=ConnectionError("flaky"))
    stream = InMemoryOrderEventStream()
    orch = _build_orchestrator(feed=feed, order_stream=stream)

    result = await orch.run()

    await _cancel_consumer(result)


# 4. Bootstrap exhausted raises


async def test_bootstrap_exhausted_raises() -> None:
    """Feed bootstrap fails all attempts → exception propagates."""

    class PermanentError(Exception):
        pass

    class AlwaysFailFeed(MarketDataFeedPort):
        bootstrapped = False
        started = False

        async def bootstrap(self) -> None:
            raise PermanentError("permanent")

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            pass

        async def ensure_fresh(self) -> None:
            pass

    orch = _build_orchestrator(feed=AlwaysFailFeed())

    with pytest.raises(PermanentError):
        await orch.run()


# 5. Stale orders cancelled


async def test_stale_orders_cancelled() -> None:
    """Seed stale orders → cancelled_orders > 0."""
    feed = FakeExchangeDataFeed()
    open_order = FakeOpenOrderAdapter()
    open_order.seed(
        "BTCUSDC",
        [
            ExchangeOpenOrder(
                exchange_order_id="EX1",
                client_order_id="CL1",
                symbol="BTCUSDC",
                side=OrderSide.BUY,
                position_side=PositionSide.LONG,
                status=OrderStatus.NEW,
                original_quantity=Decimal("0.1"),
            ),
        ],
    )
    audit = CapturingAuditLog()
    orch = _build_orchestrator(
        feed=feed,
        open_order_port=open_order,
        order_stream=InMemoryOrderEventStream(),
        audit_log=audit,
    )

    result = await orch.run()

    assert result.cancelled_orders == 1
    assert len(open_order.cancel_calls) == 1
    await _cancel_consumer(result)


# 6. Divergence with orphaned_exchange → UNCERTAIN blocks start


async def test_divergence_uncertain_blocks_start() -> None:
    """Orphaned exchange position → UNCERTAIN → no consumer_task."""
    feed = FakeExchangeDataFeed()
    account_state = FakeAccountState()
    instrument_id = InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="BTCUSDC",
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )
    account_state.set_snapshot(
        "test",
        AccountSnapshot(
            account="test",
            balances=(),
            positions=(
                ExchangePositionLeg(
                    instrument_id=instrument_id,
                    position_side=PositionSide.LONG,
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("100"),
                ),
            ),
            total_wallet_balance=Decimal("10000"),
            available_balance=Decimal("5000"),
        ),
    )
    # DB has no legs → orphaned_exchange → UNCERTAIN
    orch = _build_orchestrator(
        feed=feed,
        account_state=account_state,
        order_stream=InMemoryOrderEventStream(),
    )

    result = await orch.run()

    assert result.health.health is ReconciliationHealth.UNCERTAIN
    assert result.consumer_task is None
    assert result.periodic_reconciliation is None


# 7. Divergence with mismatches only → DEGRADED continues


async def test_divergence_degraded_continues() -> None:
    """Quantity mismatch → DEGRADED → continues with consumer."""
    feed = FakeExchangeDataFeed()
    account_state = FakeAccountState()
    instrument_repo = FakeInstrumentRepo()
    position_leg_repo = FakePositionLegRepo()

    instrument_id = InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="BTCUSDC",
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )

    # Seed instrument
    from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter, InstrumentInfo

    db_id = await instrument_repo.upsert_instrument(
        InstrumentInfo(
            instrument_id=instrument_id,
            status="TRADING",
            filters=InstrumentFilter(
                tick_size=Decimal("0.01"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
            ),
            price_precision=2,
            quantity_precision=3,
            base_asset_precision=8,
            quote_asset_precision=8,
        )
    )

    # Exchange has 1.0 BTC — no orphaned_exchange, just a qty mismatch → DEGRADED
    account_state.set_snapshot(
        "test",
        AccountSnapshot(
            account="test",
            balances=(),
            positions=(
                ExchangePositionLeg(
                    instrument_id=instrument_id,
                    position_side=PositionSide.LONG,
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("100"),
                ),
            ),
            total_wallet_balance=Decimal("10000"),
            available_balance=Decimal("5000"),
        ),
    )

    # DB has 0.5 BTC → mismatch → DEGRADED
    from dojiwick.domain.models.value_objects.position_leg import PositionLeg

    clock = FixedClock()
    await position_leg_repo.insert_leg(
        PositionLeg(
            account="test",
            instrument_id=db_id,
            position_side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            opened_at=clock.now_utc(),
        )
    )

    stream = InMemoryOrderEventStream()
    orch = _build_orchestrator(
        feed=feed,
        account_state=account_state,
        instrument_repo=instrument_repo,
        position_leg_repo=position_leg_repo,
        order_stream=stream,
        clock=clock,
    )

    result = await orch.run()

    assert result.health.health is ReconciliationHealth.DEGRADED
    assert result.consumer_task is not None
    assert result.periodic_reconciliation is not None

    await _cancel_consumer(result)


# 8. Replay missed events


async def test_replay_missed_events() -> None:
    """Seed cursor + events + order report → replayed_events > 0, order_id resolved."""
    feed = FakeExchangeDataFeed()
    cursor_repo = FakeStreamCursorRepo()
    cursor_repo.cursors["in_memory"] = StreamCursorRecord(
        stream_name="in_memory",
        last_event_id="EX1",
        last_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    stream = InMemoryOrderEventStream()
    event = OrderEvent(
        order_id=0,
        event_type=OrderEventType.FILLED,
        occurred_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        exchange_order_id="EX2",
    )
    stream.push_event(event)

    # Seed an order report so replay can resolve the order_id
    order_report_repo = FakeOrderReportRepo()
    await order_report_repo.upsert_report(
        OrderReport(
            order_request_id=42,
            exchange_order_id="EX2",
            status=OrderStatus.FILLED,
        )
    )

    order_event_repo = FakeOrderEventRepository()
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        cursor_repo=cursor_repo,
        order_event_repo=order_event_repo,
        order_report_repo=order_report_repo,
    )

    result = await orch.run()

    assert result.replayed_events == 1
    assert len(order_event_repo.events) == 1
    assert order_event_repo.events[0].order_id == 42

    await _cancel_consumer(result)


# 9. No cursor → replay skipped


async def test_replay_no_cursor_skips() -> None:
    """No cursor in repo → replayed_events = 0."""
    feed = FakeExchangeDataFeed()
    stream = InMemoryOrderEventStream()
    stream.push_event(
        OrderEvent(
            order_id=1,
            event_type=OrderEventType.FILLED,
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    order_event_repo = FakeOrderEventRepository()
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        order_event_repo=order_event_repo,
    )

    result = await orch.run()

    assert result.replayed_events == 0
    assert len(order_event_repo.events) == 0

    await _cancel_consumer(result)


# 10. Re-reconciliation after replay


async def test_re_reconciliation_after_replay() -> None:
    """Replay events → reconciliation runs twice (verify via audit log)."""
    feed = FakeExchangeDataFeed()
    cursor_repo = FakeStreamCursorRepo()
    cursor_repo.cursors["in_memory"] = StreamCursorRecord(
        stream_name="in_memory",
        last_event_id="EX1",
        last_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    stream = InMemoryOrderEventStream()
    stream.push_event(
        OrderEvent(
            order_id=0,
            event_type=OrderEventType.FILLED,
            occurred_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            exchange_order_id="EX2",
        )
    )
    # Seed order report so replay resolves the event
    order_report_repo = FakeOrderReportRepo()
    await order_report_repo.upsert_report(
        OrderReport(
            order_request_id=1,
            exchange_order_id="EX2",
            status=OrderStatus.FILLED,
        )
    )
    audit = CapturingAuditLog()
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        cursor_repo=cursor_repo,
        order_report_repo=order_report_repo,
        audit_log=audit,
    )

    result = await orch.run()

    assert result.replayed_events == 1
    # Two reconciliation_startup audit events (initial + post-replay)
    recon_events = [e for e in audit.events if e["event_type"] == "reconciliation_startup"]
    assert len(recon_events) == 2

    await _cancel_consumer(result)


# 11. Consumer task started


async def test_consumer_task_started() -> None:
    """order_stream present → consumer_task is not None."""
    feed = FakeExchangeDataFeed()
    stream = InMemoryOrderEventStream()
    orch = _build_orchestrator(feed=feed, order_stream=stream)

    result = await orch.run()

    assert result.consumer_task is not None
    assert not result.consumer_task.done()

    await _cancel_consumer(result)


# 12. Stale recon result after replay (1A)


async def test_stale_recon_result_after_replay() -> None:
    """Re-recon after replay should replace stale pre-replay result."""
    from dojiwick.domain.models.value_objects.reconciliation import (
        PositionMismatch,
        ReconciliationResult,
    )

    _CLEAN = ReconciliationResult(
        orphaned_exchange=(),
        orphaned_db=(),
        mismatches=(),
    )
    _DEGRADED = ReconciliationResult(
        orphaned_exchange=(),
        orphaned_db=(),
        mismatches=(
            PositionMismatch(
                pair="BTCUSDC",
                order_id="",
                db_state="0.5",
                exchange_state="1.0",
            ),
        ),
    )

    class ResolvedAfterReplayReconciliation:
        """Returns divergence on first call, clean on second."""

        def __init__(self) -> None:
            self._call_count = 0

        async def run_startup_gate(
            self,
            pair_symbols: tuple[str, ...],  # noqa: ARG002
        ) -> ReconciliationResult:
            self._call_count += 1
            if self._call_count == 1:
                return _DEGRADED
            return _CLEAN

        async def run_periodic_check(
            self,
            pair_symbols: tuple[str, ...],  # noqa: ARG002
        ) -> ReconciliationResult:
            return _CLEAN

    feed = FakeExchangeDataFeed()
    cursor_repo = FakeStreamCursorRepo()
    cursor_repo.cursors["in_memory"] = StreamCursorRecord(
        stream_name="in_memory",
        last_event_id="EX1",
        last_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    stream = InMemoryOrderEventStream()
    stream.push_event(
        OrderEvent(
            order_id=0,
            event_type=OrderEventType.FILLED,
            occurred_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            exchange_order_id="EX2",
        )
    )
    order_report_repo = FakeOrderReportRepo()
    await order_report_repo.upsert_report(
        OrderReport(
            order_request_id=1,
            exchange_order_id="EX2",
            status=OrderStatus.FILLED,
        )
    )
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        cursor_repo=cursor_repo,
        order_report_repo=order_report_repo,
        reconciliation_service_override=ResolvedAfterReplayReconciliation(),  # type: ignore[arg-type]
    )

    result = await orch.run()

    # Post-replay recon was clean → health should be NORMAL, not DEGRADED
    assert result.health.health is ReconciliationHealth.NORMAL
    await _cancel_consumer(result)


# 13. Clean startup resets persisted HALT (1B)


async def test_clean_startup_resets_persisted_halt() -> None:
    """If prior run left HALT in bot_state, clean startup resets to NORMAL."""
    from dojiwick.domain.models.entities.bot_state import BotState

    bot_state_repo = InMemoryBotStateRepo()
    await bot_state_repo.update_state(BotState(recon_health=ReconciliationHealth.HALT))

    feed = FakeExchangeDataFeed()
    stream = InMemoryOrderEventStream()
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        bot_state_repo=bot_state_repo,
    )

    result = await orch.run()

    assert result.health.health is ReconciliationHealth.NORMAL
    # Verify DB state was actually persisted as NORMAL
    persisted = await bot_state_repo.get_state()
    assert persisted.recon_health is ReconciliationHealth.NORMAL

    await _cancel_consumer(result)


# 14. Replay skips unknown exchange_order_id (1C)


async def test_replay_skips_unknown_exchange_order_id() -> None:
    """Events with no matching OrderReport are skipped during replay."""
    feed = FakeExchangeDataFeed()
    cursor_repo = FakeStreamCursorRepo()
    cursor_repo.cursors["in_memory"] = StreamCursorRecord(
        stream_name="in_memory",
        last_event_id="EX1",
        last_event_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    stream = InMemoryOrderEventStream()
    # Event with unknown exchange_order_id — no OrderReport seeded
    stream.push_event(
        OrderEvent(
            order_id=0,
            event_type=OrderEventType.FILLED,
            occurred_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            exchange_order_id="UNKNOWN_EX",
        )
    )

    order_event_repo = FakeOrderEventRepository()
    orch = _build_orchestrator(
        feed=feed,
        order_stream=stream,
        cursor_repo=cursor_repo,
        order_event_repo=order_event_repo,
    )

    result = await orch.run()

    assert result.replayed_events == 0
    assert len(order_event_repo.events) == 0

    await _cancel_consumer(result)

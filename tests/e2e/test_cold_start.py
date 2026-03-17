"""E2E test: cold-start reconcile-before-first-tick.

Validates the full startup sequence: bootstrap exchange cache via REST,
run startup reconciliation gate, then execute the first tick only after
reconciliation completes.
"""

from decimal import Decimal

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.enums import PositionMode
from dojiwick.domain.models.value_objects.account_state import AccountBalance, AccountSnapshot
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.exchange.cache import ExchangeCache
from dojiwick.infrastructure.exchange.cached_context_provider import CachedContextProvider
from dojiwick.infrastructure.exchange.feed import ExchangeDataFeed, FeedStatus
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.audit_log import CapturingAuditLog
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.market_data_provider import InMemoryMarketDataProvider
from fixtures.fakes.order_event_stream import InMemoryOrderEventStream
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.reconciliation import CleanReconciliation
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository


def _empty_account(account: str = "default") -> AccountSnapshot:
    return AccountSnapshot(
        account=account,
        balances=(
            AccountBalance(
                asset="USDC",
                wallet_balance=Decimal(10_000),
                available_balance=Decimal(5_000),
            ),
        ),
        positions=(),
        total_wallet_balance=Decimal(10_000),
        available_balance=Decimal(5_000),
    )


async def test_cold_start_reconcile_then_tick() -> None:
    """Full cold-start sequence: bootstrap -> reconcile -> first tick.

    Verifies:
    1. Exchange cache is populated via REST bootstrap.
    2. Reconciliation gate completes before first tick.
    3. Reconciliation results are persisted for observability.
    4. First tick executes successfully using cached context.
    """
    pairs = ("BTCUSDC", "ETHUSDC")

    # --- Arrange: wire up the system ---
    clock = SystemClock()
    cache = ExchangeCache(clock=clock)
    market_data = InMemoryMarketDataProvider()
    market_data.set_prices({"BTCUSDC": 96_000.0, "ETHUSDC": 3_400.0})

    account_state = FakeAccountState()
    account_state.set_snapshot("default", _empty_account())

    order_stream = InMemoryOrderEventStream()

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=order_stream,
        pairs=pairs,
        account="default",
    )

    audit = CapturingAuditLog()
    recon_service = ReconciliationService(
        reconciliation_port=CleanReconciliation(),
        audit_log=audit,
    )

    # --- Step 1: Bootstrap exchange cache via REST ---
    await feed.bootstrap()
    assert cache.has_data
    assert feed.status == FeedStatus.BOOTSTRAPPING

    # --- Step 2: Start feed (WS or REST fallback) ---
    await feed.start()
    assert feed.status in (FeedStatus.WS_ACTIVE, FeedStatus.REST_FALLBACK)

    # --- Step 3: Startup reconciliation gate (blocks until complete) ---
    recon_result = await recon_service.run_startup_gate(pairs)
    assert recon_result.is_clean

    # Reconciliation results persisted for observability
    assert len(audit.events) == 1
    assert audit.events[0]["event_type"] == "reconciliation_startup"

    # --- Step 4: First tick executes using cached context ---
    context_provider = CachedContextProvider(cache=cache)
    outcomes_repo = CapturingOutcomeRepo()

    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)

    _s = default_settings()
    service = TickService(
        settings=_s,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=context_provider,
        execution_gateway=DryRunGateway(),
        outcome_repository=outcomes_repo,
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=NullVetoService(),
        execution_planner=planner,
        account_state_provider=account_state,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(_s),
    )

    outcomes = await service.run_tick(pairs)

    assert len(outcomes) == len(pairs)
    assert len(outcomes_repo.outcomes) == len(pairs)

    # --- Step 5: Clean shutdown ---
    await feed.stop()
    assert feed.status == FeedStatus.DISCONNECTED

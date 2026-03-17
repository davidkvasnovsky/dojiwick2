"""Integration tests for reconciliation health-aware tick execution."""

from datetime import UTC, datetime

import pytest

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.enums import DecisionStatus, PositionMode, ReconciliationHealth
from dojiwick.domain.errors import CircuitBreakerTrippedError
from dojiwick.domain.models.entities.bot_state import BotState
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import CapturingGateway, DryRunGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository

from fixtures.factories.integration import empty_snapshot, signal_triggering_context_builder


def _make_tick_service(
    bot_state: BotState | None = None,
    execution_gateway: CapturingGateway | DryRunGateway | None = None,
) -> tuple[TickService, InMemoryBotStateRepo, CapturingOutcomeRepo]:
    context = signal_triggering_context_builder().build()
    outcomes_repo = CapturingOutcomeRepo()
    if execution_gateway is None:
        execution_gateway = DryRunGateway()

    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    bot_repo = InMemoryBotStateRepo(_state=bot_state or BotState())

    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=execution_gateway,
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=outcomes_repo,
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        bot_state_repository=bot_repo,
        veto_service=NullVetoService(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )
    return service, bot_repo, outcomes_repo


_PAIRS = ("BTC/USDC", "ETH/USDC")


async def test_halt_prevents_execution() -> None:
    """BotState with HALT raises CircuitBreakerTrippedError."""
    state = BotState(recon_health=ReconciliationHealth.HALT)
    service, _, _ = _make_tick_service(bot_state=state)

    with pytest.raises(CircuitBreakerTrippedError, match="HALT"):
        await service.run_tick(pairs=_PAIRS)


async def test_tick_skips_frozen_symbols() -> None:
    """BotState with UNCERTAIN + frozen BTC/USDC masks that pair."""
    state = BotState(
        recon_health=ReconciliationHealth.UNCERTAIN,
        recon_health_since=datetime(2026, 1, 1, tzinfo=UTC),
        recon_frozen_symbols=("BTC/USDC",),
    )
    gateway = CapturingGateway()
    service, _, _ = _make_tick_service(bot_state=state, execution_gateway=gateway)

    outcomes = await service.run_tick(pairs=_PAIRS)

    assert len(outcomes) == 2
    # BTC/USDC should be masked (hold/skipped), ETH/USDC may execute
    btc_outcome = next(o for o in outcomes if o.pair == "BTC/USDC")
    assert btc_outcome.status in (
        DecisionStatus.HOLD,
        DecisionStatus.BLOCKED_RISK,
        DecisionStatus.VETOED,
        DecisionStatus.ERROR,
    )


async def test_degraded_with_empty_frozen_processes_normally() -> None:
    """BotState with DEGRADED but no frozen symbols processes all pairs."""
    state = BotState(
        recon_health=ReconciliationHealth.DEGRADED,
        recon_health_since=datetime(2026, 1, 1, tzinfo=UTC),
        recon_frozen_symbols=(),
    )
    service, _, _ = _make_tick_service(bot_state=state)

    outcomes = await service.run_tick(pairs=_PAIRS)

    assert len(outcomes) == 2
    assert any(o.status is DecisionStatus.EXECUTED for o in outcomes)


async def test_degraded_freezes_symbols() -> None:
    """BotState with DEGRADED + frozen BTC/USDC masks that pair."""
    state = BotState(
        recon_health=ReconciliationHealth.DEGRADED,
        recon_health_since=datetime(2026, 1, 1, tzinfo=UTC),
        recon_frozen_symbols=("BTC/USDC",),
    )
    gateway = CapturingGateway()
    service, _, _ = _make_tick_service(bot_state=state, execution_gateway=gateway)

    outcomes = await service.run_tick(pairs=_PAIRS)

    assert len(outcomes) == 2
    btc_outcome = next(o for o in outcomes if o.pair == "BTC/USDC")
    assert btc_outcome.status in (
        DecisionStatus.HOLD,
        DecisionStatus.BLOCKED_RISK,
        DecisionStatus.VETOED,
        DecisionStatus.ERROR,
    )

"""Integration tests for feature flags wired through the full tick pipeline."""

import pytest

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings
from dojiwick.domain.enums import DecisionAuthority, PositionMode
from dojiwick.domain.errors import CircuitBreakerTrippedError
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.system.clock import SystemClock
from dojiwick.domain.contracts.policies.veto import VetoServicePort
from fixtures.factories.infrastructure import SettingsBuilder
from fixtures.factories.integration import empty_snapshot, signal_triggering_context_builder
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository


def _make_tick_service(
    settings_builder: SettingsBuilder,
    veto_service: VetoServicePort | None = None,
) -> tuple[TickService, CapturingOutcomeRepo]:
    context = signal_triggering_context_builder().build()
    outcomes_repo = CapturingOutcomeRepo()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    built_settings = settings_builder.build()
    service = TickService(
        settings=built_settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=outcomes_repo,
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=veto_service,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(built_settings),
    )
    return service, outcomes_repo


async def test_global_halt_raises_circuit_breaker() -> None:
    builder = SettingsBuilder().with_global_halt()
    service, _ = _make_tick_service(builder, veto_service=NullVetoService())

    with pytest.raises(CircuitBreakerTrippedError, match="global_halt"):
        await service.run_tick()


async def test_disable_llm_produces_deterministic_authority() -> None:
    builder = SettingsBuilder().with_disable_llm().with_ai_veto(enabled=True, veto_enabled=True)
    service, _ = _make_tick_service(builder, veto_service=NullVetoService())

    outcomes = await service.run_tick()

    assert len(outcomes) > 0
    for outcome in outcomes:
        assert outcome.authority == DecisionAuthority.DETERMINISTIC_ONLY


async def test_exits_only_blocks_new_entries() -> None:
    builder = SettingsBuilder().with_exits_only()
    service, _ = _make_tick_service(builder, veto_service=NullVetoService())

    outcomes = await service.run_tick()

    assert len(outcomes) > 0
    for outcome in outcomes:
        assert outcome.status.value in ("hold", "executed")
        # In exits_only mode, all non-HOLD candidates are masked out.
        # The signal_triggering_context_builder produces BUY signals,
        # which should be masked → HOLD outcomes.
        if outcome.action.value == 1:  # BUY
            assert outcome.status.value == "hold"

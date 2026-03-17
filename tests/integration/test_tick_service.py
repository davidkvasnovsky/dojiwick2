"""Tick service integration tests."""

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.enums import PositionMode
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.system.clock import SystemClock
from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.contracts.policies.veto import VetoServicePort
from fixtures.fakes.account_state import FakeAccountState
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import CapturingGateway, DryRunGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository
from fixtures.fakes.veto import RejectFirstVeto

from fixtures.factories.integration import empty_snapshot, signal_triggering_context_builder


def _make_tick_service(
    context: BatchDecisionContext | None = None,
    veto_service: VetoServicePort | None = None,
    outcomes_repo: CapturingOutcomeRepo | None = None,
    execution_gateway: ExecutionGatewayPort | None = None,
) -> tuple[TickService, CapturingOutcomeRepo]:
    """Build a TickService with required planner/account state wired."""
    if context is None:
        context = signal_triggering_context_builder().build()
    if outcomes_repo is None:
        outcomes_repo = CapturingOutcomeRepo()
    if execution_gateway is None:
        execution_gateway = DryRunGateway()

    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

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
        veto_service=veto_service,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )
    return service, outcomes_repo


async def test_tick_service_executes_with_null_veto() -> None:
    context = signal_triggering_context_builder().build()
    service, outcomes_repo = _make_tick_service(context=context, veto_service=NullVetoService())

    outcomes = await service.run_tick(context.market.pairs)

    assert len(outcomes) == context.size
    assert any(item.status.value == "executed" for item in outcomes)
    assert len(outcomes_repo.outcomes) == context.size


async def test_tick_service_marks_vetoed_rows() -> None:
    context = signal_triggering_context_builder().build()
    service, _ = _make_tick_service(context=context, veto_service=RejectFirstVeto())

    outcomes = await service.run_tick(context.market.pairs)

    assert outcomes[0].status.value == "vetoed"


async def test_tick_service_with_planner_path() -> None:
    """TickService uses planner for execution."""
    context = signal_triggering_context_builder().build()
    service, outcomes_repo = _make_tick_service(context=context, veto_service=NullVetoService())

    outcomes = await service.run_tick(context.market.pairs)

    assert len(outcomes) == context.size
    assert any(item.status.value == "executed" for item in outcomes)
    assert len(outcomes_repo.outcomes) == context.size


async def test_tick_service_passes_tick_id_to_gateway() -> None:
    """execute_plan receives the tick_id kwarg from the tick pipeline."""
    context = signal_triggering_context_builder().build()
    gateway = CapturingGateway()
    service, _ = _make_tick_service(context=context, veto_service=NullVetoService(), execution_gateway=gateway)

    await service.run_tick(context.market.pairs)

    assert len(gateway.calls) == 1
    assert gateway.calls[0].tick_id != ""

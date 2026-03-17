"""Integration tests — tick service metrics instrumentation."""

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.enums import PositionMode
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.system.clock import SystemClock
from dojiwick.domain.contracts.gateways.context_provider import ContextProviderPort
from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.contracts.policies.veto import VetoServicePort
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from fixtures.fakes.account_state import FakeAccountState
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.models.value_objects.submission_ack import SubmissionAck
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.metrics import CapturingMetrics
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository
from fixtures.fakes.veto import RejectAllVeto

import pytest

from fixtures.factories.integration import empty_snapshot, signal_triggering_context_builder


class _RaisingGateway(ExecutionGatewayPort):
    """Raises RuntimeError on execute_plan."""

    async def execute_plan(self, plan: ExecutionPlan, *, tick_id: str = "") -> tuple[ExecutionReceipt, ...]:
        del plan, tick_id
        raise RuntimeError("gateway explosion")

    async def cancel_order(self, pair: str, order_id: str) -> SubmissionAck:
        del pair, order_id
        raise RuntimeError("gateway explosion")

    async def place_order(self, *args: object, **kwargs: object) -> SubmissionAck:
        del args, kwargs
        raise RuntimeError("gateway explosion")


def _make_tick_service(
    *,
    context: BatchDecisionContext | None = None,
    veto_service: VetoServicePort | None = None,
    metrics: CapturingMetrics | None = None,
    context_provider: ContextProviderPort | None = None,
) -> tuple[TickService, CapturingMetrics]:
    """Build a TickService wired with capturing metrics."""
    if context is None:
        context = signal_triggering_context_builder().build()
    if metrics is None:
        metrics = CapturingMetrics()

    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    provider = context_provider or StaticBatchContextProvider(context=context)

    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=provider,
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=veto_service,
        metrics=metrics,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )
    return service, metrics


async def test_tick_emits_duration_metric() -> None:
    """Successful tick emits tick_duration_seconds observation."""
    service, metrics = _make_tick_service(veto_service=NullVetoService())
    context = signal_triggering_context_builder().build()

    await service.run_tick(context.market.pairs)

    assert "tick_duration_seconds" in metrics.observations
    assert metrics.observations["tick_duration_seconds"][0] > 0


async def test_tick_emits_failure_metric() -> None:
    """Failed tick increments tick_failure_total."""
    metrics = CapturingMetrics()
    context = signal_triggering_context_builder().build()

    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=_RaisingGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=NullVetoService(),
        metrics=metrics,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

    with pytest.raises(RuntimeError, match="gateway explosion"):
        await service.run_tick(context.market.pairs)

    assert metrics.counters.get("tick_failure_total", 0) == 1


async def test_veto_block_metric() -> None:
    """Vetoed candidates increment ai_veto_block_total."""
    service, metrics = _make_tick_service(veto_service=RejectAllVeto())
    context = signal_triggering_context_builder().build()

    await service.run_tick(context.market.pairs)

    assert metrics.counters.get("ai_veto_block_total", 0) > 0


async def test_risk_block_metric() -> None:
    """Risk-blocked candidates increment risk_block_total."""
    from dataclasses import replace

    import numpy as np

    settings = default_settings().model_copy(
        update={"risk": default_settings().risk.model_copy(update={"max_open_positions": 1})}
    )

    metrics = CapturingMetrics()
    # Build context with open_positions_total=1 so max_positions rule blocks
    ctx_builder = signal_triggering_context_builder()
    context = ctx_builder.build()
    # Override open_positions_total to be at the limit
    portfolio = replace(
        context.portfolio,
        open_positions_total=np.ones(context.size, dtype=np.int64),
    )
    context = replace(context, portfolio=portfolio)

    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    service = TickService(
        settings=settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(settings.risk),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=NullVetoService(),
        metrics=metrics,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

    await service.run_tick(context.market.pairs)

    assert metrics.counters.get("risk_block_total", 0) > 0

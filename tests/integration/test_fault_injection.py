"""Fault injection integration tests — resilience under component failures."""

from datetime import UTC, datetime

import pytest

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.config.schema import Settings
from fixtures.factories.infrastructure import default_instrument_map, default_settings
from dojiwick.domain.enums import DecisionAuthority, DecisionStatus, PositionMode, TickStatus
from dojiwick.domain.errors import PostExecutionPersistenceError
from dojiwick.domain.hashing import compute_tick_id
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.infrastructure import SettingsBuilder
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway, ErrorGateway, RejectAllGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo, FailingOutcomeRepo
from fixtures.fakes.regime_classifier import TimeoutClassifier
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import FailingTickRepo, InMemoryTickRepo
from fixtures.fakes.veto import FailVeto, TimeoutVeto

from fixtures.factories.integration import empty_snapshot


def _make_service(
    *,
    settings: Settings | None = None,
    tick_repo: InMemoryTickRepo | None = None,
    execution_gateway: DryRunGateway | ErrorGateway | RejectAllGateway | None = None,
    outcome_repository: CapturingOutcomeRepo | FailingOutcomeRepo | None = None,
    veto_service: TimeoutVeto | FailVeto | None = None,
    regime_classifier: TimeoutClassifier | None = None,
) -> tuple[TickService, InMemoryTickRepo]:
    s = settings or default_settings()
    context = ContextBuilder().trending_up().build()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())
    repo = tick_repo or InMemoryTickRepo()
    return (
        TickService(
            settings=s,
            strategy_registry=build_default_strategy_registry(),
            risk_engine=build_default_risk_engine(s.risk),
            clock=SystemClock(),
            context_provider=StaticBatchContextProvider(context=context),
            execution_gateway=execution_gateway or DryRunGateway(),
            execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
            account_state_provider=account_state,
            outcome_repository=outcome_repository or CapturingOutcomeRepo(),
            regime_repository=InMemoryRegimeRepo(),
            tick_repository=repo,
            bot_state_repository=InMemoryBotStateRepo(),
            veto_service=veto_service,
            regime_classifier=regime_classifier,
            target_ids=("btc_usdc", "eth_usdc"),
            instrument_map=default_instrument_map(s),
        ),
        repo,
    )


# LLM failure tests


async def test_veto_timeout_fail_open() -> None:
    """TimeoutVeto + fail_open=True → tick completes, trades execute."""
    settings = SettingsBuilder().with_ai_veto(enabled=True, veto_enabled=True, fail_open_on_error=True).build()
    service, repo = _make_service(settings=settings, veto_service=TimeoutVeto())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = settings.trading.active_pairs

    outcomes = await service.run_tick(pairs=pairs, at=fixed_time)

    assert len(outcomes) > 0
    assert any(o.status == DecisionStatus.EXECUTED for o in outcomes)

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.COMPLETED


async def test_veto_timeout_fail_closed() -> None:
    """TimeoutVeto + fail_open=False → tick completes, all candidates vetoed."""
    settings = SettingsBuilder().with_ai_veto(enabled=True, veto_enabled=True, fail_open_on_error=False).build()
    service, _repo = _make_service(settings=settings, veto_service=TimeoutVeto())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    outcomes = await service.run_tick(pairs=settings.trading.active_pairs, at=fixed_time)

    assert len(outcomes) > 0
    # All candidates should be vetoed — none executed
    assert not any(o.status == DecisionStatus.EXECUTED for o in outcomes)


async def test_veto_unexpected_error_always_blocks() -> None:
    """FailVeto(RuntimeError) → all candidates vetoed regardless of fail_open."""
    settings = SettingsBuilder().with_ai_veto(enabled=True, veto_enabled=True, fail_open_on_error=True).build()
    service, _repo = _make_service(settings=settings, veto_service=FailVeto(RuntimeError))
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    outcomes = await service.run_tick(pairs=settings.trading.active_pairs, at=fixed_time)

    assert len(outcomes) > 0
    assert not any(o.status == DecisionStatus.EXECUTED for o in outcomes)


async def test_regime_classifier_timeout_fail_open() -> None:
    """TimeoutClassifier + regime_fail_open=True → tick completes, authority = DETERMINISTIC_ONLY."""
    settings = SettingsBuilder().with_ai_regime(enabled=True, fail_open_on_error=True).build()
    outcome_repo = CapturingOutcomeRepo()
    service, _repo = _make_service(
        settings=settings, regime_classifier=TimeoutClassifier(), outcome_repository=outcome_repo
    )
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    outcomes = await service.run_tick(pairs=settings.trading.active_pairs, at=fixed_time)

    assert len(outcomes) > 0
    assert all(o.authority == DecisionAuthority.DETERMINISTIC_ONLY for o in outcomes)


# Persistence failure tests


async def test_outcome_persistence_failure_raises() -> None:
    """FailingOutcomeRepo → PostExecutionPersistenceError raised."""
    service, _repo = _make_service(outcome_repository=FailingOutcomeRepo())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    with pytest.raises(PostExecutionPersistenceError):
        await service.run_tick(pairs=service.settings.trading.active_pairs, at=fixed_time)


async def test_tick_repo_insert_failure_propagates() -> None:
    """FailingTickRepo → error propagates (tick_repo is not best-effort at insert level)."""
    context = ContextBuilder().trending_up().build()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())
    _s = default_settings()
    service = TickService(
        settings=_s,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(_s.risk),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=CapturingOutcomeRepo(),
        regime_repository=InMemoryRegimeRepo(),
        tick_repository=FailingTickRepo(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(_s),
    )
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    with pytest.raises(RuntimeError, match="tick repo failure"):
        await service.run_tick(pairs=service.settings.trading.active_pairs, at=fixed_time)


# Execution failure tests


async def test_execution_gateway_error_produces_error_receipts() -> None:
    """ErrorGateway → outcomes reflect error status, tick still completes."""
    outcome_repo = CapturingOutcomeRepo()
    service, tick_repo = _make_service(execution_gateway=ErrorGateway(), outcome_repository=outcome_repo)
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    outcomes = await service.run_tick(pairs=pairs, at=fixed_time)

    assert len(outcomes) > 0

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = tick_repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.COMPLETED


async def test_reject_all_gateway_produces_rejected_outcomes() -> None:
    """RejectAllGateway → outcomes reflect rejection."""
    outcome_repo = CapturingOutcomeRepo()
    service, _repo = _make_service(execution_gateway=RejectAllGateway(), outcome_repository=outcome_repo)
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    outcomes = await service.run_tick(pairs=service.settings.trading.active_pairs, at=fixed_time)

    assert len(outcomes) > 0
    # No outcome should be EXECUTED since all were rejected
    assert not any(o.status == DecisionStatus.EXECUTED for o in outcomes)

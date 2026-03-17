"""Integration test — atomic post-execution persistence with UoW rollback."""

from datetime import UTC, datetime

import pytest

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from fixtures.factories.infrastructure import default_instrument_map, default_settings
from dojiwick.domain.enums import PositionMode, TickStatus
from dojiwick.domain.errors import PostExecutionPersistenceError
from dojiwick.domain.hashing import compute_tick_id
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.integration import empty_snapshot
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.outcome_repository import FailingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import InMemoryTickRepo
from fixtures.fakes.unit_of_work import FakeUnitOfWork


async def test_atomic_rollback_on_persistence_failure() -> None:
    """When outcome persistence fails inside UoW, the transaction rolls back
    and tick status is never updated to COMPLETED."""
    context = ContextBuilder().trending_up().build()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())

    uow = FakeUnitOfWork()
    tick_repo = InMemoryTickRepo()

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
        outcome_repository=FailingOutcomeRepo(),
        regime_repository=InMemoryRegimeRepo(),
        tick_repository=tick_repo,
        bot_state_repository=InMemoryBotStateRepo(),
        unit_of_work=uow,
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(_s),
    )

    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    with pytest.raises(PostExecutionPersistenceError):
        await service.run_tick(pairs=pairs, at=fixed_time)

    assert uow.rolled_back == 1
    assert uow.committed == 0

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = tick_repo.get(tick_id)
    assert record is not None
    assert record.status != TickStatus.COMPLETED

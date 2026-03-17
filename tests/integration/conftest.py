"""Integration test configuration — auto-marks all tests as integration."""

import pytest

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.config.schema import Settings
from fixtures.factories.infrastructure import (
    default_instrument_map,
    default_risk_settings,
    default_settings as _default_settings,
)
from dojiwick.domain.enums import PositionMode
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.integration import empty_snapshot
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import InMemoryTickRepo


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.integration)


@pytest.fixture
def default_settings() -> Settings:
    """Default engine settings."""
    return _default_settings()


@pytest.fixture
def tick_service(default_settings: Settings) -> TickService:
    """TickService with fakes pre-wired."""
    context = ContextBuilder().trending_up().build()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())
    return TickService(
        settings=default_settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=account_state,
        outcome_repository=CapturingOutcomeRepo(),
        regime_repository=InMemoryRegimeRepo(),
        tick_repository=InMemoryTickRepo(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

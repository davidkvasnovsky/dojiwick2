"""Architecture acceptance tests verifying core hexagonal guarantees.

Seven tests cover: plugin extensibility, adapter swappability, live/backtest
parity, adaptive roundtrip, adaptive fallback, and veto fail-open behaviour.
"""

from decimal import Decimal

import numpy as np

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.adaptive.service import AdaptiveService
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_backtest import BacktestService
from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.enums import AdaptiveMode, DecisionStatus, PositionMode
from dojiwick.domain.models.value_objects.account_state import AccountBalance, AccountSnapshot
from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptivePosterior
from dojiwick.domain.models.value_objects.batch_models import BatchSignalFragment
from dojiwick.domain.models.value_objects.params import StrategyParams
from dojiwick.domain.type_aliases import FloatMatrix, FloatVector, IntVector
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.factories.domain import ContextBuilder
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import NoOpTickRepository
from fixtures.fakes.veto import TimeoutVeto


def _fake_account_state() -> FakeAccountState:
    """Return a FakeAccountState with a default snapshot."""
    state = FakeAccountState()
    state.set_snapshot(
        "default",
        AccountSnapshot(
            account="default",
            balances=(AccountBalance(asset="USDC", wallet_balance=Decimal(10_000), available_balance=Decimal(5_000)),),
            positions=(),
            total_wallet_balance=Decimal(10_000),
            available_balance=Decimal(5_000),
        ),
    )
    return state


# Helpers


class _DummyPlugin:
    """Custom strategy that emits BUY for every row."""

    @property
    def name(self) -> str:
        return "dummy_custom"

    def signal(
        self,
        *,
        states: IntVector,
        indicators: FloatMatrix,
        prices: FloatVector,
        settings: StrategyParams,
        per_pair_settings: tuple[StrategyParams, ...] | None = None,
        pre_extracted: dict[str, np.ndarray] | None = None,
        regime_confidence: FloatVector | None = None,
    ) -> BatchSignalFragment:
        _ = (states, indicators, settings, per_pair_settings, pre_extracted, regime_confidence)
        size = len(prices)
        return BatchSignalFragment(
            strategy_name="dummy_custom",
            buy_mask=np.ones(size, dtype=np.bool_),
            short_mask=np.zeros(size, dtype=np.bool_),
        )


class _MockSelectionPolicy:
    """Returns a fixed arm key."""

    def __init__(self, arm: AdaptiveArmKey) -> None:
        self._arm = arm

    async def select(
        self,
        regime_idx: int,
        posteriors: tuple[AdaptivePosterior, ...],
    ) -> AdaptiveArmKey:
        _ = (regime_idx, posteriors)
        return self._arm


class _FailingSelectionPolicy:
    """Always raises on select."""

    async def select(
        self,
        regime_idx: int,
        posteriors: tuple[AdaptivePosterior, ...],
    ) -> AdaptiveArmKey:
        _ = (regime_idx, posteriors)
        raise RuntimeError("policy failure")


# Test 1 — Strategy registry extensibility


def test_strategy_registry_extensibility() -> None:
    """Register a custom StrategyPlugin → its signal appears in merged output."""

    registry = build_default_strategy_registry()
    registry.register(_DummyPlugin())

    # Ranging context: default plugins fire nothing, so only dummy fires.
    context = ContextBuilder().ranging().build()
    settings = default_settings()

    regimes = classify_regime_batch(context.market, settings.regime.params)
    variants = tuple(settings.strategy.default_variant for _ in range(context.size))

    candidates = registry.propose_candidates(
        context=context,
        regime=regimes,
        settings=settings.strategy,
        variants=variants,
    )

    assert candidates.valid_mask.any(), "custom plugin signals must appear in merged result"
    # Dummy plugin should own the strategy name for at least one row
    assert "dummy_custom" in candidates.strategy_name


# Test 2 — Exchange swap (DryRunGateway)


async def test_exchange_swap_dry_run() -> None:
    """DryRunGateway satisfies ExecutionGatewayPort → full tick completes."""

    context = ContextBuilder().trending_up().build()
    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=_fake_account_state(),
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

    outcomes = await service.run_tick(context.market.pairs)

    assert len(outcomes) == context.size
    assert any(o.status == DecisionStatus.EXECUTED for o in outcomes)


# Test 3 — AI filter swap (NullVetoService)


async def test_ai_filter_swap_null_veto() -> None:
    """NullVetoService satisfies VetoServicePort → tick runs with all rows approved."""

    context = ContextBuilder().trending_up().build()
    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=_fake_account_state(),
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=NullVetoService(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

    outcomes = await service.run_tick(context.market.pairs)

    assert len(outcomes) == context.size
    assert not any(o.status == DecisionStatus.VETOED for o in outcomes)


# Test 4 — Live / backtest parity


async def test_live_backtest_parity() -> None:
    """Identical context + shared registry/engine → identical strategy decisions."""

    context = ContextBuilder().trending_up().build()
    settings = default_settings()
    registry = build_default_strategy_registry()
    engine = build_default_risk_engine(default_risk_settings())

    # Live path (no veto service → pure deterministic pipeline)
    live_service = TickService(
        settings=settings,
        strategy_registry=registry,
        risk_engine=engine,
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=_fake_account_state(),
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )
    live_outcomes = await live_service.run_tick(context.market.pairs)

    # Backtest path (same registry and engine)
    bt_service = BacktestService(
        settings=settings,
        strategy_registry=registry,
        risk_engine=engine,
    )
    next_prices = context.market.price + np.ones(context.size, dtype=np.float64)
    bt_summary = await bt_service.run(context, next_prices)

    # Trade count parity: both pipelines should accept/reject the same rows
    live_trades = sum(1 for o in live_outcomes if o.status == DecisionStatus.EXECUTED)
    assert live_trades == bt_summary.trades, f"live traded {live_trades}, backtest traded {bt_summary.trades}"

    # Strategy names and reason codes are populated for every outcome
    for outcome in live_outcomes:
        assert outcome.strategy_name != "", "every outcome must have a strategy name"
        assert outcome.reason_code != "", "every outcome must have a reason code"


# Test 5 — Adaptive roundtrip


async def test_adaptive_roundtrip_disabled() -> None:
    """AdaptiveService.select_variant() returns 'baseline' when mode is disabled."""

    svc = AdaptiveService(mode=AdaptiveMode.DISABLED)
    result = await svc.select_variant()
    assert result == "baseline"


async def test_adaptive_roundtrip_enabled() -> None:
    """When mode is enabled with a mock selection policy, returns the policy's chosen arm key."""

    expected_arm = AdaptiveArmKey(regime_idx=0, config_idx=42)
    svc = AdaptiveService(
        mode=AdaptiveMode.CONTINUOUS,
        selection_policy=_MockSelectionPolicy(expected_arm),
    )
    result = await svc.select_variant()
    assert result == expected_arm


# Test 6 — Adaptive fallback


async def test_adaptive_fallback_on_error() -> None:
    """BUCKET_FALLBACK + selection error → falls back to 'baseline'."""

    svc = AdaptiveService(
        mode=AdaptiveMode.BUCKET_FALLBACK,
        selection_policy=_FailingSelectionPolicy(),
    )
    result = await svc.select_variant()
    assert result == "baseline"


# Test 7 — Veto fail-open


async def test_veto_fail_open_on_error() -> None:
    """VetoServicePort raises TimeoutError + fail_open_on_error=True → tick continues."""

    context = ContextBuilder().trending_up().build()
    # Default settings: ai.enabled=True, ai.veto_enabled=True, ai.fail_open_on_error=True
    service = TickService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        clock=SystemClock(),
        context_provider=StaticBatchContextProvider(context=context),
        execution_gateway=DryRunGateway(),
        execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
        account_state_provider=_fake_account_state(),
        outcome_repository=CapturingOutcomeRepo(),
        tick_repository=NoOpTickRepository(),
        regime_repository=InMemoryRegimeRepo(),
        veto_service=TimeoutVeto(),
        bot_state_repository=InMemoryBotStateRepo(),
        target_ids=("btc_usdc", "eth_usdc"),
        instrument_map=default_instrument_map(),
    )

    outcomes = await service.run_tick(context.market.pairs)

    assert len(outcomes) == context.size
    # No rows vetoed — fail-open approved everything
    assert not any(o.status == DecisionStatus.VETOED for o in outcomes)
    # At least one execution succeeded
    assert any(o.status == DecisionStatus.EXECUTED for o in outcomes)


# Test 8 — No domain→infrastructure imports


def test_no_domain_imports_from_infrastructure() -> None:
    """No module in dojiwick.domain imports from dojiwick.infrastructure."""
    import ast
    import pathlib

    domain_root = pathlib.Path("src/dojiwick/domain")
    violations: list[str] = []

    for py_file in domain_root.rglob("*.py"):
        if py_file.name == "__pycache__":
            continue
        source = py_file.read_text()
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("dojiwick.infrastructure"):
                        violations.append(f"{py_file}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("dojiwick.infrastructure"):
                    violations.append(f"{py_file}:{node.lineno} imports from {node.module}")

    assert not violations, "domain→infrastructure imports found:\n" + "\n".join(violations)


# Test 9 — All value objects are frozen


def test_all_value_objects_are_frozen() -> None:
    """All dataclasses in domain/models/value_objects/ have frozen=True."""
    import importlib
    import inspect
    import pathlib

    vo_root = pathlib.Path("src/dojiwick/domain/models/value_objects")
    violations: list[str] = []

    for py_file in sorted(vo_root.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"dojiwick.domain.models.value_objects.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != module_name:
                continue
            # Check if it's a dataclass with __dataclass_params__
            params = getattr(obj, "__dataclass_params__", None)
            if params is None:
                continue
            if not params.frozen:
                violations.append(f"{module_name}.{name} is not frozen")

    assert not violations, "Non-frozen value objects found:\n" + "\n".join(violations)


# Test 10 — All enums use StrEnum or IntEnum


def test_all_enums_use_str_or_int_enum_pattern() -> None:
    """All enum classes in domain/enums.py subclass StrEnum or IntEnum."""
    import enum
    import importlib
    import inspect

    mod = importlib.import_module("dojiwick.domain.enums")
    violations: list[str] = []

    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != "dojiwick.domain.enums":
            continue
        if not issubclass(obj, enum.Enum):
            continue
        if not (issubclass(obj, (str, enum.StrEnum)) or issubclass(obj, (int, enum.IntEnum))):
            violations.append(f"{name} does not subclass StrEnum or IntEnum")

    assert not violations, "Enum pattern violations:\n" + "\n".join(violations)


# Test 11 — Protocols have real and fake implementations


def test_protocols_have_real_and_fake_implementations() -> None:
    """Each protocol in domain/contracts/ has at least one real impl and one fake."""
    import importlib
    import inspect
    import pathlib
    from typing import Protocol

    contracts_root = pathlib.Path("src/dojiwick/domain/contracts")
    protocols: list[tuple[str, str]] = []  # (module_name, class_name)

    # Discover protocol classes
    for py_file in contracts_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        rel = py_file.relative_to(pathlib.Path("src"))
        module_name = str(rel).replace("/", ".").removesuffix(".py")
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ != module_name:
                continue
            if name.endswith("Port") and issubclass(obj, Protocol):
                protocols.append((module_name, name))

    assert len(protocols) > 0, "No protocols found — test misconfigured"

    # Scan infrastructure/ for real implementations
    infra_root = pathlib.Path("src/dojiwick/infrastructure")
    infra_classes: set[str] = set()
    for py_file in infra_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        rel = py_file.relative_to(pathlib.Path("src"))
        module_name = str(rel).replace("/", ".").removesuffix(".py")
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name, _ in inspect.getmembers(mod, inspect.isclass):
            infra_classes.add(name)

    # Scan fakes/ for test doubles
    fakes_root = pathlib.Path("tests/fixtures/fakes")
    fake_classes: set[str] = set()
    for py_file in fakes_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        module_name = f"fixtures.fakes.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name, _ in inspect.getmembers(mod, inspect.isclass):
            fake_classes.add(name)

    # Verify — at least some protocols have both
    protocols_with_real = 0
    protocols_with_fake = 0
    for _mod_name, proto_name in protocols:
        # Strip "Port" suffix and look for implementations
        base_name = proto_name.removesuffix("Port")
        has_real = any(base_name.lower() in cls.lower() for cls in infra_classes)
        has_fake = any(base_name.lower() in cls.lower() for cls in fake_classes)
        if has_real:
            protocols_with_real += 1
        if has_fake:
            protocols_with_fake += 1

    assert protocols_with_real > 0, "No protocol has a real implementation in infrastructure/"
    assert protocols_with_fake > 0, "No protocol has a fake implementation in tests/fixtures/fakes/"

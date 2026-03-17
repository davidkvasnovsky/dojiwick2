"""Integration tests for deterministic tick replay and deduplication."""

from datetime import UTC, datetime

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.config.fingerprint import fingerprint_settings
from dojiwick.config.schema import Settings
from fixtures.factories.infrastructure import default_instrument_map, default_risk_settings, default_settings
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from dojiwick.domain.enums import PositionMode, TickStatus
from dojiwick.domain.hashing import compute_tick_id
from dojiwick.infrastructure.system.clock import SystemClock
from fixtures.factories.domain import ContextBuilder
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.bot_state_repository import InMemoryBotStateRepo
from fixtures.fakes.context_provider import StaticBatchContextProvider
from fixtures.fakes.execution import DryRunGateway, ErrorGateway
from fixtures.fakes.outcome_repository import CapturingOutcomeRepo, FailingOutcomeRepo
from fixtures.fakes.regime_repository import InMemoryRegimeRepo
from fixtures.fakes.tick_repository import InMemoryTickRepo

from fixtures.factories.integration import empty_snapshot


def _make_service(
    *,
    settings: Settings | None = None,
    context: BatchDecisionContext | None = None,
    tick_repo: InMemoryTickRepo | None = None,
    execution_gateway: DryRunGateway | ErrorGateway | None = None,
    outcome_repository: CapturingOutcomeRepo | FailingOutcomeRepo | None = None,
) -> tuple[TickService, InMemoryTickRepo]:
    settings = settings or default_settings()
    context = context or ContextBuilder().trending_up().build()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", empty_snapshot())
    repo = tick_repo or InMemoryTickRepo()
    return (
        TickService(
            settings=settings,
            strategy_registry=build_default_strategy_registry(),
            risk_engine=build_default_risk_engine(settings.risk),
            clock=SystemClock(),
            context_provider=StaticBatchContextProvider(context=context),
            execution_gateway=execution_gateway or DryRunGateway(),
            execution_planner=DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY),
            account_state_provider=account_state,
            outcome_repository=outcome_repository or CapturingOutcomeRepo(),
            regime_repository=InMemoryRegimeRepo(),
            tick_repository=repo,
            bot_state_repository=InMemoryBotStateRepo(),
            target_ids=("btc_usdc", "eth_usdc"),
            instrument_map=default_instrument_map(settings),
        ),
        repo,
    )


async def test_same_context_produces_same_tick_id() -> None:
    """Two runs with the same context and time produce the same tick_id."""
    service, _repo = _make_service()
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    expected_id = compute_tick_id(service.config_hash, fixed_time, pairs)

    outcomes = await service.run_tick(pairs=pairs, at=fixed_time)
    assert len(outcomes) > 0
    assert all(o.tick_id == expected_id for o in outcomes)


async def test_dedup_prevents_double_execution() -> None:
    """Second run_tick with the same tick_id returns empty tuple (dedup)."""
    service, _repo = _make_service()
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    first = await service.run_tick(pairs=pairs, at=fixed_time)
    assert len(first) > 0

    second = await service.run_tick(pairs=pairs, at=fixed_time)
    assert second == ()


async def test_tick_record_persisted_on_success() -> None:
    """After a successful tick, TickRecord has COMPLETED status."""
    service, repo = _make_service()
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    await service.run_tick(pairs=pairs, at=fixed_time)

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.COMPLETED
    assert record.duration_ms is not None
    assert record.duration_ms >= 0
    assert record.inputs_hash != ""
    assert record.intent_hash != ""


async def test_tick_record_persisted_on_failure() -> None:
    """After a failed tick, TickRecord has FAILED status with error_message."""
    service, repo = _make_service(outcome_repository=FailingOutcomeRepo())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    try:
        await service.run_tick(pairs=pairs, at=fixed_time)
    except Exception:
        pass

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.FAILED
    assert record.error_message is not None
    assert record.duration_ms is not None


# Phase 7 edge-case tests


async def test_tick_record_hashes_deterministic_across_services() -> None:
    """Two fresh TickService instances produce the same intent_hash, ops_hash, inputs_hash."""
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    context = ContextBuilder().trending_up().with_observed_at(fixed_time).build()

    service_a, repo_a = _make_service(context=context)
    service_b, repo_b = _make_service(context=context)
    pairs = service_a.settings.trading.active_pairs

    await service_a.run_tick(pairs=pairs, at=fixed_time)
    await service_b.run_tick(pairs=pairs, at=fixed_time)

    tick_id = compute_tick_id(service_a.config_hash, fixed_time, pairs)
    rec_a = repo_a.get(tick_id)
    rec_b = repo_b.get(tick_id)

    assert rec_a is not None and rec_b is not None
    assert rec_a.inputs_hash == rec_b.inputs_hash
    assert rec_a.intent_hash == rec_b.intent_hash
    assert rec_a.ops_hash == rec_b.ops_hash


async def test_different_times_produce_different_tick_ids() -> None:
    """Two different observation times produce different tick_ids."""
    service, _repo = _make_service()
    pairs = service.settings.trading.active_pairs

    t1 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2024, 6, 15, 12, 1, 0, tzinfo=UTC)

    id_1 = compute_tick_id(service.config_hash, t1, pairs)
    id_2 = compute_tick_id(service.config_hash, t2, pairs)

    assert id_1 != id_2


async def test_config_change_produces_different_config_hash() -> None:
    """Modified risk settings produce a different config_hash."""
    settings_a = default_settings()
    hash_a = fingerprint_settings(settings_a).trading_sha256
    service_a, _repo_a = _make_service(settings=settings_a)
    service_a.config_hash = hash_a

    settings_b = default_settings().model_copy(update={"risk": default_risk_settings(max_daily_loss_pct=0.99)})
    hash_b = fingerprint_settings(settings_b).trading_sha256
    service_b, _repo_b = _make_service(settings=settings_b)
    service_b.config_hash = hash_b

    assert service_a.config_hash != service_b.config_hash


async def test_dedup_after_failure_allows_retry() -> None:
    """FAILED tick can be retried (InMemoryTickRepo re-inserts over FAILED)."""
    repo = InMemoryTickRepo()

    # First run fails (FailingOutcomeRepo)
    service_fail, _ = _make_service(tick_repo=repo, outcome_repository=FailingOutcomeRepo())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service_fail.settings.trading.active_pairs

    try:
        await service_fail.run_tick(pairs=pairs, at=fixed_time)
    except Exception:
        pass

    tick_id = compute_tick_id(service_fail.config_hash, fixed_time, pairs)
    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.FAILED

    # Retry with working outcome repo — same tick_id should succeed
    service_ok, _ = _make_service(tick_repo=repo)
    outcomes = await service_ok.run_tick(pairs=pairs, at=fixed_time)
    assert len(outcomes) > 0

    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.COMPLETED


async def test_execution_error_gateway_still_completes_tick() -> None:
    """ErrorGateway produces error receipts but tick record is still COMPLETED."""
    service, repo = _make_service(execution_gateway=ErrorGateway())
    fixed_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
    pairs = service.settings.trading.active_pairs

    outcomes = await service.run_tick(pairs=pairs, at=fixed_time)
    assert len(outcomes) > 0

    tick_id = compute_tick_id(service.config_hash, fixed_time, pairs)
    record = repo.get(tick_id)
    assert record is not None
    assert record.status == TickStatus.COMPLETED

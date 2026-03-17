"""Root test fixtures for dojiwick engine."""

import pytest

from fixtures.factories.compute import DecisionOutcomeBuilder, ExecutionReceiptBuilder
from fixtures.factories.domain import (
    AIEvaluationResultBuilder,
    BotStateBuilder,
    CandleBuilder,
    ContextBuilder,
    InstrumentIdBuilder,
    PairTradingStateBuilder,
    PositionLegKeyBuilder,
    SignalBuilder,
    TargetLegPositionBuilder,
    TimeSeriesBuilder,
)
from fixtures.factories.infrastructure import PerformanceSnapshotBuilder, SettingsBuilder

from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext


@pytest.fixture
def sample_context() -> BatchDecisionContext:
    """Aligned two-pair context with deterministic signal-friendly indicators."""

    return ContextBuilder().trending_up().build()


@pytest.fixture
def context_builder() -> ContextBuilder:
    """Return a fresh ContextBuilder for fluent test setup."""

    return ContextBuilder()


@pytest.fixture
def receipt_builder() -> ExecutionReceiptBuilder:
    """Return a fresh ExecutionReceiptBuilder."""

    return ExecutionReceiptBuilder()


@pytest.fixture
def outcome_builder() -> DecisionOutcomeBuilder:
    """Return a fresh DecisionOutcomeBuilder."""

    return DecisionOutcomeBuilder()


@pytest.fixture
def candle_builder() -> CandleBuilder:
    """Return a fresh CandleBuilder."""

    return CandleBuilder()


@pytest.fixture
def signal_builder() -> SignalBuilder:
    """Return a fresh SignalBuilder."""

    return SignalBuilder()


@pytest.fixture
def bot_state_builder() -> BotStateBuilder:
    """Return a fresh BotStateBuilder."""

    return BotStateBuilder()


@pytest.fixture
def pair_state_builder() -> PairTradingStateBuilder:
    """Return a fresh PairTradingStateBuilder."""

    return PairTradingStateBuilder()


@pytest.fixture
def performance_snapshot_builder() -> PerformanceSnapshotBuilder:
    """Return a fresh PerformanceSnapshotBuilder."""

    return PerformanceSnapshotBuilder()


@pytest.fixture
def settings_builder() -> SettingsBuilder:
    """Return a fresh SettingsBuilder."""

    return SettingsBuilder()


@pytest.fixture
def instrument_id_builder() -> InstrumentIdBuilder:
    """Return a fresh InstrumentIdBuilder (Binance USD-M BTC/USDC defaults)."""

    return InstrumentIdBuilder()


@pytest.fixture
def position_leg_key_builder() -> PositionLegKeyBuilder:
    """Return a fresh PositionLegKeyBuilder."""

    return PositionLegKeyBuilder()


@pytest.fixture
def target_leg_position_builder() -> TargetLegPositionBuilder:
    """Return a fresh TargetLegPositionBuilder."""

    return TargetLegPositionBuilder()


@pytest.fixture
def ai_evaluation_result_builder() -> AIEvaluationResultBuilder:
    """Return a fresh AIEvaluationResultBuilder."""

    return AIEvaluationResultBuilder()


@pytest.fixture
def time_series_builder() -> TimeSeriesBuilder:
    """Return a fresh TimeSeriesBuilder."""

    return TimeSeriesBuilder()

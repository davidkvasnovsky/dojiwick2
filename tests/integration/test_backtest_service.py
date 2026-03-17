"""Backtest service tests."""

import numpy as np

from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_backtest import BacktestService
from dojiwick.config.schema import Settings
from fixtures.factories.infrastructure import default_regime_settings, default_risk_settings, default_settings
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from fixtures.factories.domain import TimeSeriesBuilder


def _service(settings: Settings | None = None) -> BacktestService:
    s = settings or default_settings()
    return BacktestService(
        settings=s,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        config_hash="test_config_hash",
    )


async def test_backtest_service_returns_summary(sample_context: BatchDecisionContext) -> None:
    service = _service()
    next_prices = sample_context.market.price + np.array([1.0, -0.5], dtype=np.float64)

    summary = await service.run(sample_context, next_prices)

    assert summary.trades >= 1
    assert isinstance(summary.total_pnl_usd, float)
    assert summary.max_drawdown_pct >= 0.0


async def test_backtest_with_hysteresis_returns_summary() -> None:
    """Smoke test: 5-bar trending_up series produces a valid summary."""
    series = TimeSeriesBuilder(n_bars=5).build()
    service = _service()

    result = await service.run_with_hysteresis(series)
    summary = result.summary

    assert summary.trades >= 0
    assert isinstance(summary.total_pnl_usd, float)
    assert isinstance(summary.sharpe_like, float)
    assert summary.max_drawdown_pct >= 0.0
    assert summary.config_hash != ""


async def test_hysteresis_suppresses_single_bar_flip() -> None:
    """A single ranging bar amid trending_up is suppressed by hysteresis_bars=3.

    Verifies at the pipeline level that the shared RegimeHysteresis instance
    inside ``run_with_hysteresis`` holds the regime stable across bars.
    """
    from dojiwick.application.orchestration.decision_pipeline import PipelineResult, run_decision_pipeline
    from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis

    series = (
        TimeSeriesBuilder(n_bars=5)
        .with_regime_sequence(["trending_up", "trending_up", "ranging", "trending_up", "trending_up"])
        .build()
    )
    registry = build_default_strategy_registry()
    risk_engine = build_default_risk_engine(default_risk_settings())

    # hysteresis_bars=3: single ranging bar at index 2 should be suppressed
    settings_3 = default_settings().model_copy(update={"regime": default_regime_settings(hysteresis_bars=3)})
    hyst_suppress = RegimeHysteresis()
    regimes_suppressed: list[np.ndarray] = []
    for ctx in series.contexts:
        pipeline: PipelineResult = await run_decision_pipeline(
            context=ctx,
            settings=settings_3,
            strategy_registry=registry,
            risk_engine=risk_engine,
            hysteresis=hyst_suppress,
        )
        regimes_suppressed.append(pipeline.regimes.coarse_state.copy())

    # hysteresis_bars=1: regime flip is instant
    settings_1 = default_settings().model_copy(update={"regime": default_regime_settings(hysteresis_bars=1)})
    hyst_instant = RegimeHysteresis()
    regimes_instant: list[np.ndarray] = []
    for ctx in series.contexts:
        pipeline = await run_decision_pipeline(
            context=ctx,
            settings=settings_1,
            strategy_registry=registry,
            risk_engine=risk_engine,
            hysteresis=hyst_instant,
        )
        regimes_instant.append(pipeline.regimes.coarse_state.copy())

    # Bar 2: suppressed keeps trending_up, instant flips to ranging
    np.testing.assert_array_equal(regimes_suppressed[0], regimes_suppressed[2])
    assert not np.array_equal(regimes_instant[0], regimes_instant[2])

    # End-to-end: run_with_hysteresis completes successfully
    service = _service()
    bt_result = await service.run_with_hysteresis(series, hysteresis_bars=3)
    assert bt_result.summary.trades >= 0
    assert bt_result.summary.config_hash != ""


async def test_hysteresis_bars_1_matches_single_bar() -> None:
    """With hysteresis_bars=1 and a 1-bar series, result matches run() on same context."""
    series = TimeSeriesBuilder(n_bars=1).build()
    service = _service()

    bt_result = await service.run_with_hysteresis(series, hysteresis_bars=1)
    result_single = await service.run(series.contexts[0], series.next_prices[0])

    assert bt_result.summary.trades == result_single.trades
    np.testing.assert_allclose(bt_result.summary.total_pnl_usd, result_single.total_pnl_usd, atol=1e-10)
    np.testing.assert_allclose(bt_result.summary.win_rate, result_single.win_rate, atol=1e-10)

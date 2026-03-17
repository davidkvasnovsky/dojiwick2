"""Optimization objective tests."""

import numpy as np
import pytest

from dojiwick.application.use_cases.optimization.objective import (
    _base_score as base_score_fn,  # pyright: ignore[reportPrivateUsage]
    _regularization as regularization_fn,  # pyright: ignore[reportPrivateUsage]
    VectorObjective,
    WalkForwardObjective,
)
from dojiwick.application.use_cases.optimization.search_space import ParamSet, extract_regularization_baseline
from dojiwick.config.param_tuning import apply_params
from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary
from dojiwick.config.scope import (
    ScopeSelector,
    StrategyOverrideValues,
    StrategyScopeResolver,
    StrategyScopeRule,
)
from dojiwick.domain.enums import ObjectiveMode
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from dojiwick.domain.models.value_objects.params import StrategyParams
from fixtures.factories.domain import TimeSeriesBuilder
from fixtures.factories.infrastructure import default_optimization_settings, default_settings, default_strategy_params


def _full_params(**overrides: float | int) -> ParamSet:
    """Return a complete ParamSet with defaults, applying overrides."""
    defaults: ParamSet = {
        "stop_atr_mult": 2.5,
        "rr_ratio": 2.5,
        "min_stop_distance_pct": 0.5,
        "mean_rsi_oversold": 20.0,
        "mean_rsi_overbought": 85.0,
        "vol_extreme_oversold": 28.0,
        "vol_extreme_overbought": 85.0,
        "trend_pullback_rsi_max": 42.0,
        "trailing_stop_activation_rr": 1.5,
        "trailing_stop_atr_mult": 1.8,
        "breakeven_after_rr": 1.0,
        "max_hold_bars": 48,
        "adx_trend_min": 22.0,
        "atr_high_pct": 0.9,
        "min_confidence": 0.55,
        "ema_spread_weak_bps": 8.0,
        "atr_low_pct": 0.30,
        "min_volume_ratio": 1.0,
        "trend_max_regime_confidence": 0.92,
    }
    defaults.update(overrides)
    return defaults


def _config_strategy() -> StrategyParams:
    """Strategy params matching config.toml values."""
    return default_strategy_params(
        stop_atr_mult=2.5,
        rr_ratio=2.5,
        min_stop_distance_pct=0.5,
        mean_rsi_oversold=20.0,
        mean_rsi_overbought=85.0,
        vol_extreme_oversold=28.0,
        vol_extreme_overbought=85.0,
        trend_pullback_rsi_max=42.0,
        trend_breakout_adx_min=35.0,
        trailing_stop_activation_rr=1.5,
        trailing_stop_atr_mult=1.8,
        breakeven_after_rr=1.0,
        min_volume_ratio=1.0,
        max_hold_bars=48,
        trend_max_regime_confidence=0.92,
    )


def _make_apply_tuned(settings: object) -> object:
    """Build apply_tuned callable for test objectives."""
    from dojiwick.config.schema import Settings

    s = settings
    assert isinstance(s, Settings)

    def _apply(params: ParamSet) -> Settings:
        return apply_params(s, params, baseline=s)

    return _apply


async def test_vector_objective_returns_float(sample_context: BatchDecisionContext) -> None:
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    objective = VectorObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        base_context=sample_context,
        next_prices=sample_context.market.price + np.array([1.0, -0.2], dtype=np.float64),
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    score = await objective.evaluate(
        {
            "stop_atr_mult": 2.5,
            "rr_ratio": 2.0,
            "min_stop_distance_pct": 0.5,
            "mean_rsi_oversold": 35.0,
            "mean_rsi_overbought": 70.0,
            "vol_extreme_oversold": 30.0,
            "vol_extreme_overbought": 75.0,
            "trend_pullback_rsi_max": 45.0,
            "adx_trend_min": 20.0,
            "atr_high_pct": 0.9,
            "min_confidence": 0.55,
            "ema_spread_weak_bps": 8.0,
            "atr_low_pct": 0.30,
            "trailing_stop_activation_rr": 1.0,
            "trailing_stop_atr_mult": 1.0,
            "breakeven_after_rr": 1.0,
            "max_hold_bars": 48,
            "min_volume_ratio": 0.8,
            "trend_max_regime_confidence": 0.90,
        }
    )

    assert isinstance(score, float)


def test_apply_params_scales_scope_rules() -> None:
    """Scope rule overrides scale proportionally when baseline is provided."""
    scope_rule = StrategyScopeRule(
        id="mean_revert_exits",
        priority=100,
        selector=ScopeSelector(strategy="mean_revert"),
        values=StrategyOverrideValues(stop_atr_mult=1.0, max_hold_bars=12),
    )
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "strategy_scope": StrategyScopeResolver(rules=(scope_rule,)),
        }
    )

    # Optimizer samples stop_atr_mult=2.0 (was 2.5) and max_hold_bars=24 (was 48)
    params = _full_params(stop_atr_mult=2.0, max_hold_bars=24)
    result = apply_params(settings, params, baseline=settings)

    scaled = result.strategy_scope.rules[0].values
    # stop_atr_mult: 1.0 * (2.0/2.5) = 0.8, clamped to mean_revert floor 1.5
    assert scaled.stop_atr_mult == pytest.approx(1.5)  # pyright: ignore[reportUnknownMemberType]
    # max_hold_bars: round(12 * (24/48)) = 6, but clamped to mean_revert floor 15
    assert scaled.max_hold_bars == 15


def test_apply_params_without_baseline_preserves_scope() -> None:
    """Without baseline, scope rules are unchanged (backward compat)."""
    scope_rule = StrategyScopeRule(
        id="mean_revert_exits",
        priority=100,
        selector=ScopeSelector(strategy="mean_revert"),
        values=StrategyOverrideValues(stop_atr_mult=1.0),
    )
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "strategy_scope": StrategyScopeResolver(rules=(scope_rule,)),
        }
    )

    result = apply_params(settings, _full_params())
    assert result.strategy_scope.rules[0].values.stop_atr_mult == 1.0


def test_regularization_uses_loaded_settings() -> None:
    """Regularization baseline comes from settings, not hardcoded defaults."""
    # Optimization requires all optional strategy fields to be explicitly set.
    strategy = default_strategy_params(
        trailing_stop_activation_rr=1.5,
        trailing_stop_atr_mult=1.0,
        breakeven_after_rr=1.0,
        max_hold_bars=48,
        trend_max_regime_confidence=0.92,
    )
    settings = default_settings().model_copy(update={"strategy": strategy})
    baseline = extract_regularization_baseline(settings)
    # Pydantic defaults differ from old hardcoded _PARAM_DEFAULTS
    assert baseline["stop_atr_mult"] == 1.5  # Pydantic default, not old 2.5
    assert baseline["rr_ratio"] == 2.0  # Pydantic default, not old 2.5

    # When params match baseline exactly, regularization penalty is zero
    params: ParamSet = {k: v for k, v in baseline.items()}
    summary = BacktestSummary(
        trades=0, total_pnl_usd=0.0, win_rate=0.0, expectancy_usd=0.0, sharpe_like=0.0, max_drawdown_pct=0.0
    )
    base = base_score_fn(settings.optimization, summary, equity_start=0.0)
    reg = settings.optimization.objective_regularization_strength * regularization_fn(params, baseline)
    assert base - reg == 0.0


def test_base_score_penalty_below_min_trades() -> None:
    """_base_score returns graduated penalty when trades < objective_min_trades."""
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "optimization": default_optimization_settings(objective_min_trades=10),
        }
    )
    summary = BacktestSummary(
        trades=5,
        total_pnl_usd=0.0,
        win_rate=0.6,
        expectancy_usd=0.0,
        sharpe_like=2.0,
        max_drawdown_pct=10.0,
        sortino=0.5,
        profit_factor=1.5,
    )
    score = base_score_fn(settings.optimization, summary, equity_start=0.0)
    min_trades_penalty = settings.optimization.objective_min_trades_penalty
    # Graduated: -2.0 - 18.0 * (5/10) = -11.0
    assert score < 0
    assert score >= min_trades_penalty
    # Zero trades should hit floor
    summary_zero = BacktestSummary(
        trades=0,
        total_pnl_usd=0.0,
        win_rate=0.6,
        expectancy_usd=0.0,
        sharpe_like=2.0,
        max_drawdown_pct=10.0,
        sortino=0.5,
        profit_factor=1.5,
    )
    score_zero = base_score_fn(settings.optimization, summary_zero, equity_start=0.0)
    assert score_zero == min_trades_penalty


def test_base_score_normal_above_min_trades() -> None:
    """_base_score returns normal composite when trades >= objective_min_trades."""
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "optimization": default_optimization_settings(objective_min_trades=10),
        }
    )
    summary = BacktestSummary(
        trades=10,
        total_pnl_usd=0.0,
        win_rate=0.6,
        expectancy_usd=0.0,
        sharpe_like=2.0,
        max_drawdown_pct=10.0,
        sortino=0.5,
        profit_factor=1.5,
    )
    score = base_score_fn(settings.optimization, summary, equity_start=0.0)
    min_trades_penalty = settings.optimization.objective_min_trades_penalty
    assert score != min_trades_penalty
    assert score > min_trades_penalty


def test_base_score_no_penalty_when_disabled() -> None:
    """_base_score with default objective_min_trades=0 applies no penalty even with 1 trade."""
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    assert settings.optimization.objective_min_trades == 0
    summary = BacktestSummary(
        trades=1, total_pnl_usd=0.0, win_rate=0.5, expectancy_usd=0.0, sharpe_like=0.0, max_drawdown_pct=0.0
    )
    score = base_score_fn(settings.optimization, summary, equity_start=0.0)
    # No penalty -- should be the normal composite (win_rate_weight * 0.5)
    assert score >= 0.0


def test_base_score_profit_factor_capped_at_5() -> None:
    """Profit factor contribution is capped at 5.0."""
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    summary_5 = BacktestSummary(
        trades=20,
        total_pnl_usd=0.0,
        win_rate=0.5,
        expectancy_usd=0.0,
        sharpe_like=1.0,
        max_drawdown_pct=10.0,
        sortino=1.0,
        profit_factor=5.0,
    )
    summary_10 = BacktestSummary(
        trades=20,
        total_pnl_usd=0.0,
        win_rate=0.5,
        expectancy_usd=0.0,
        sharpe_like=1.0,
        max_drawdown_pct=10.0,
        sortino=1.0,
        profit_factor=10.0,
    )
    score_at_5 = base_score_fn(settings.optimization, summary_5, equity_start=0.0)
    score_at_10 = base_score_fn(settings.optimization, summary_10, equity_start=0.0)
    assert score_at_5 == score_at_10


# WalkForwardObjective tests


async def test_wfo_returns_float() -> None:
    """WalkForwardObjective.evaluate returns a float score."""
    series = TimeSeriesBuilder(n_bars=10).build()
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    objective = WalkForwardObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        series=series,
        n_folds=2,
        consistency_penalty=0.5,
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    score = await objective.evaluate(_full_params())
    assert isinstance(score, float)


def test_wfo_fold_count() -> None:
    """_fold_series has correct number of folds with expected bar counts."""
    series = TimeSeriesBuilder(n_bars=10).build()
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    objective = WalkForwardObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        series=series,
        n_folds=5,
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    assert len(objective._fold_series) == 5  # pyright: ignore[reportPrivateUsage]
    # Each fold should have 2 bars (10 / 5)
    for fold in objective._fold_series:  # pyright: ignore[reportPrivateUsage]
        assert fold.n_bars == 2


def test_wfo_fold_count_uneven() -> None:
    """Last fold absorbs remainder when bars don't divide evenly."""
    series = TimeSeriesBuilder(n_bars=11).build()
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    objective = WalkForwardObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        series=series,
        n_folds=3,
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    folds = objective._fold_series  # pyright: ignore[reportPrivateUsage]
    assert len(folds) == 3
    # 11 // 3 = 3; first two folds get 3 bars, last fold gets remainder (5)
    assert folds[0].n_bars == 3
    assert folds[1].n_bars == 3
    assert folds[2].n_bars == 5


def test_wfo_min_trades_scaling() -> None:
    """_per_fold_min_trades correctly scaled from objective_min_trades."""
    series = TimeSeriesBuilder(n_bars=10).build()
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "optimization": default_optimization_settings(objective_min_trades=20),
        }
    )
    objective = WalkForwardObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        series=series,
        n_folds=5,
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    # 20 // 5 = 4
    assert objective._per_fold_min_trades == 4  # pyright: ignore[reportPrivateUsage]


def test_wfo_min_trades_scaling_floor() -> None:
    """_per_fold_min_trades floors to 1 even when objective_min_trades < n_folds."""
    series = TimeSeriesBuilder(n_bars=10).build()
    settings = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "optimization": default_optimization_settings(objective_min_trades=2),
        }
    )
    objective = WalkForwardObjective(
        settings=settings,
        apply_tuned=_make_apply_tuned(settings),  # pyright: ignore[reportArgumentType]
        series=series,
        n_folds=5,
        target_ids=("btc_usdc", "eth_usdc"),
        venue="binance",
        product="usd_c",
    )

    # max(1, 2 // 5) = max(1, 0) = 1
    assert objective._per_fold_min_trades == 1  # pyright: ignore[reportPrivateUsage]


def test_objective_mode_enum() -> None:
    """ObjectiveMode values parse correctly from strings."""
    assert ObjectiveMode("is_oos") == ObjectiveMode.IS_OOS
    assert ObjectiveMode("walk_forward") == ObjectiveMode.WALK_FORWARD
    # Settings default
    settings = default_settings().model_copy(update={"strategy": _config_strategy()})
    assert settings.optimization.objective_mode == ObjectiveMode.IS_OOS
    # Explicit walk_forward
    settings_wf = default_settings().model_copy(
        update={
            "strategy": _config_strategy(),
            "optimization": default_optimization_settings(objective_mode=ObjectiveMode.WALK_FORWARD),
        }
    )
    assert settings_wf.optimization.objective_mode == ObjectiveMode.WALK_FORWARD

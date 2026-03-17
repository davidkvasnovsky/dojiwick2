"""Tests for Gaps 17-20: benchmark, warmup, strategy selection, monthly PnL."""

import numpy as np
import pytest

from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import StrategyRegistry, build_default_strategy_registry
from dojiwick.application.use_cases.run_backtest import BacktestService
from dojiwick.config.schema import Settings
from fixtures.factories.infrastructure import (
    default_backtest_settings,
    default_risk_settings,
    default_settings,
    default_trading_settings,
)
from fixtures.factories.domain import TimeSeriesBuilder


def _service(settings: Settings | None = None) -> BacktestService:
    s = settings or default_settings()
    return BacktestService(
        settings=s,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        config_hash="test_config_hash",
    )


# ── Gap 17: Buy-and-Hold Benchmark ──────────────────────────────────────


async def test_benchmark_pnl_reflects_price_change() -> None:
    """Benchmark PnL should be proportional to (final - initial) / initial."""
    series = TimeSeriesBuilder(n_bars=3, pairs=("BTC/USDC",)).with_price_deltas([[5.0], [5.0], [5.0]]).build()
    service = _service()
    result = await service.run_with_hysteresis(series, hysteresis_bars=1)
    summary = result.summary

    initial = float(series.contexts[0].market.price[0])
    final = float(series.next_prices[-1][0])
    equity = float(series.contexts[0].portfolio.equity_usd[0])
    expected_benchmark = ((final / initial) - 1.0) * equity

    np.testing.assert_allclose(summary.benchmark_pnl_usd, expected_benchmark, rtol=1e-6)


async def test_benchmark_pnl_is_zero_when_prices_unchanged() -> None:
    """When final price equals initial price, benchmark PnL should be ~0."""
    series = TimeSeriesBuilder(n_bars=3, pairs=("BTC/USDC",)).with_price_deltas([[0.0], [0.0], [0.0]]).build()
    service = _service()
    result = await service.run_with_hysteresis(series, hysteresis_bars=1)
    np.testing.assert_allclose(result.summary.benchmark_pnl_usd, 0.0, atol=1e-10)


# ── Gap 18: Configurable Warmup Bars ────────────────────────────────────


def test_warmup_bars_default() -> None:
    bs = default_backtest_settings()
    assert bs.warmup_bars == 200


def test_warmup_bars_custom() -> None:
    bs = default_backtest_settings(warmup_bars=100)
    assert bs.warmup_bars == 100


def test_warmup_bars_invalid() -> None:
    with pytest.raises(ValueError, match="warmup_bars must be >= 1"):
        default_backtest_settings(warmup_bars=0)


# ── Gap 19: Strategy Selection ──────────────────────────────────────────


def _plugin_count(registry: StrategyRegistry) -> int:
    return len(registry._plugins)  # pyright: ignore[reportPrivateUsage]


def _plugin_names(registry: StrategyRegistry) -> set[str]:
    return {p.name for p in registry._plugins}  # pyright: ignore[reportPrivateUsage]


def test_registry_all_strategies_by_default() -> None:
    registry = build_default_strategy_registry()
    assert _plugin_count(registry) == 3


def test_registry_filter_single_strategy() -> None:
    registry = build_default_strategy_registry(enabled=("trend_follow",))
    assert _plugin_count(registry) == 1
    assert _plugin_names(registry) == {"trend_follow"}


def test_registry_filter_two_strategies() -> None:
    registry = build_default_strategy_registry(enabled=("trend_follow", "mean_revert"))
    assert _plugin_count(registry) == 2
    assert _plugin_names(registry) == {"trend_follow", "mean_revert"}


def test_registry_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy names"):
        build_default_strategy_registry(enabled=("nonexistent",))


def test_enabled_strategies_config_default() -> None:
    ts = default_trading_settings()
    assert ts.enabled_strategies is None


def test_enabled_strategies_config_set() -> None:
    ts = default_trading_settings(enabled_strategies=("trend_follow", "mean_revert"))
    assert ts.enabled_strategies == ("trend_follow", "mean_revert")


# ── Gap 20: Monthly PnL ────────────────────────────────────────────────


async def test_monthly_pnl_present_in_result() -> None:
    """run_with_hysteresis should populate monthly_pnl."""
    series = TimeSeriesBuilder(n_bars=5).build()
    service = _service()
    result = await service.run_with_hysteresis(series)
    assert result.monthly_pnl is not None
    assert isinstance(result.monthly_pnl, dict)


async def test_monthly_pnl_keys_are_yyyy_mm() -> None:
    """All monthly_pnl keys should be YYYY-MM format."""
    series = TimeSeriesBuilder(n_bars=5).build()
    service = _service()
    result = await service.run_with_hysteresis(series)
    assert result.monthly_pnl is not None
    for key in result.monthly_pnl:
        assert len(key) == 7
        assert key[4] == "-"
        int(key[:4])  # valid year
        int(key[5:])  # valid month


async def test_monthly_pnl_sums_to_total() -> None:
    """Sum of monthly PnL should roughly equal total PnL."""
    series = TimeSeriesBuilder(n_bars=5).build()
    service = _service()
    result = await service.run_with_hysteresis(series)
    assert result.monthly_pnl is not None
    monthly_sum = sum(result.monthly_pnl.values())
    np.testing.assert_allclose(monthly_sum, result.summary.total_pnl_usd, atol=0.01)

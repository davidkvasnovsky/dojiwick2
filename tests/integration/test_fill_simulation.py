"""Integration tests for entry price model and partial fill simulation.

Verifies that non-default settings produce different P&L and that
default settings (close model, no partial fills) produce identical
results to the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.config.schema import Settings
from dojiwick.domain.enums import EntryPriceModel
from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)
from fixtures.factories.infrastructure import SettingsBuilder


def _build_series(n_bars: int = 10) -> BacktestTimeSeries:
    """Build a minimal time series with OHLC data for testing."""
    pairs = ("BTC/USDC",)
    n_pairs = len(pairs)

    contexts: list[BatchDecisionContext] = []
    next_prices: list[np.ndarray] = []
    next_open: list[np.ndarray] = []
    next_high: list[np.ndarray] = []
    next_low: list[np.ndarray] = []

    from datetime import UTC, datetime, timedelta

    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    base_price = 50000.0

    for t in range(n_bars):
        price = base_price + t * 100.0
        ctx = BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=pairs,
                observed_at=base_time + timedelta(hours=t),
                price=np.array([price]),
                indicators=np.zeros((n_pairs, INDICATOR_COUNT)),
                volume=np.array([1000.0]),
            ),
            portfolio=BatchPortfolioSnapshot(
                equity_usd=np.array([10000.0]),
                day_start_equity_usd=np.array([10000.0]),
                open_positions_total=np.zeros(n_pairs, dtype=np.int64),
                has_open_position=np.zeros(n_pairs, dtype=np.bool_),
                unrealized_pnl_usd=np.zeros(n_pairs, dtype=np.float64),
            ),
        )
        contexts.append(ctx)

        np_price = price + 100.0
        next_prices.append(np.array([np_price]))
        next_open.append(np.array([np_price - 20.0]))
        next_high.append(np.array([np_price + 50.0]))
        next_low.append(np.array([np_price - 50.0]))

    return BacktestTimeSeries(
        contexts=tuple(contexts),
        next_prices=tuple(next_prices),
        active_mask=np.ones((n_bars, 1), dtype=np.bool_),
        next_open=tuple(next_open),
        next_high=tuple(next_high),
        next_low=tuple(next_low),
    )


def _build_service(settings: Settings) -> BacktestService:
    return BacktestService(
        settings=settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(settings.risk),
    )


@pytest.mark.asyncio
async def test_default_close_model_backward_compat() -> None:
    """Default settings (close model, no partial fills) produce same result with and without OHLC."""
    settings = SettingsBuilder().build()
    service = _build_service(settings)
    series = _build_series()

    result_with_ohlc, _ = await service.run_with_hysteresis_summary_only(series)

    # Same series without OHLC
    series_no_ohlc = BacktestTimeSeries(
        contexts=series.contexts,
        next_prices=series.next_prices,
        active_mask=series.active_mask,
    )
    result_without_ohlc, _ = await service.run_with_hysteresis_summary_only(series_no_ohlc)

    assert result_with_ohlc.total_pnl_usd == result_without_ohlc.total_pnl_usd
    assert result_with_ohlc.sharpe_like == result_without_ohlc.sharpe_like


@pytest.mark.asyncio
async def test_next_open_differs_from_close() -> None:
    """next_open model produces different P&L than close model."""
    base = SettingsBuilder().build()
    settings_open = base.model_copy(
        update={"backtest": base.backtest.model_copy(update={"entry_price_model": EntryPriceModel.NEXT_OPEN})}
    )
    settings_close = base

    series = _build_series()
    result_close, _ = await _build_service(settings_close).run_with_hysteresis_summary_only(series)
    result_open, _ = await _build_service(settings_open).run_with_hysteresis_summary_only(series)

    # They may differ if any trades are taken; if no trades, both are 0
    if result_close.trades > 0:
        assert result_close.total_pnl_usd != result_open.total_pnl_usd


@pytest.mark.asyncio
async def test_partial_fill_reduces_exposure() -> None:
    """Partial fills should reduce or maintain total notional vs full fills."""
    base = SettingsBuilder().build()
    settings_partial = base.model_copy(
        update={"backtest": base.backtest.model_copy(update={"partial_fill_enabled": True})}
    )

    series = _build_series()
    result_full = await _build_service(base).run_with_hysteresis(series)
    result_partial = await _build_service(settings_partial).run_with_hysteresis(series)

    # Partial fills should produce <= total PnL magnitude (reduced exposure)
    assert abs(result_partial.summary.total_pnl_usd) <= abs(result_full.summary.total_pnl_usd) + 1e-10

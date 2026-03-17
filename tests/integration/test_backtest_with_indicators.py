"""Integration test: backtest with real indicator computation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np

from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.services.backtest_builder import build_backtest_time_series
from dojiwick.application.use_cases.run_backtest import BacktestService
from fixtures.factories.infrastructure import default_risk_settings, default_settings
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval


def _make_trending_candles(pair: str, n: int, base_price: float = 100.0) -> tuple[Candle, ...]:
    """Generate synthetic uptrending candles."""
    rng = np.random.default_rng(42)
    candles: list[Candle] = []
    price = base_price
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(n):
        price += rng.uniform(0.1, 0.5)
        o = Decimal(str(round(price, 2)))
        h = Decimal(str(round(price + rng.uniform(1, 3), 2)))
        low = Decimal(str(round(price - rng.uniform(0.5, 1.5), 2)))
        c = Decimal(str(round(price + rng.uniform(-0.2, 0.3), 2)))
        candles.append(
            Candle(
                pair=pair,
                interval=CandleInterval("1h"),
                open_time=t0 + timedelta(hours=i),
                open=o,
                high=h,
                low=low,
                close=c,
                volume=Decimal("1000"),
            )
        )
    return tuple(candles)


async def test_backtest_with_indicators_produces_valid_summary() -> None:
    """Build time series from synthetic candles -> run backtest -> verify summary."""
    pairs = ("BTC/USDC", "ETH/USDC")
    n_candles = 200 + 60

    candles_by_pair = {
        "BTC/USDC": _make_trending_candles("BTC/USDC", n_candles, base_price=40000.0),
        "ETH/USDC": _make_trending_candles("ETH/USDC", n_candles, base_price=2500.0),
    }

    series = build_backtest_time_series(candles_by_pair, pairs)

    assert series.n_bars >= 1
    assert series.n_pairs == 2

    # All indicators should be finite (warmup was trimmed)
    for ctx in series.contexts:
        assert np.all(np.isfinite(ctx.market.indicators))

    service = BacktestService(
        settings=default_settings(),
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )

    result = await service.run_with_hysteresis(series)

    assert result.summary.trades >= 0
    assert np.isfinite(result.summary.total_pnl_usd)
    assert np.isfinite(result.summary.win_rate)
    assert np.isfinite(result.summary.sharpe_like)
    assert result.summary.max_drawdown_pct >= 0.0


async def test_builder_trims_warmup_bars() -> None:
    """Verify that the builder trims warmup and produces correct bar count."""
    pairs = ("BTC/USDC",)
    n_candles = 260
    candles_by_pair = {"BTC/USDC": _make_trending_candles("BTC/USDC", n_candles)}

    series = build_backtest_time_series(candles_by_pair, pairs, warmup_bars=200)

    # After trimming 200 and reserving 1 bar for next_prices:
    expected_bars = n_candles - 200 - 1
    assert series.n_bars == expected_bars

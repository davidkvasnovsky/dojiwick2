"""Validation tests for BacktestTimeSeries."""

import numpy as np
import pytest

from dojiwick.application.use_cases.run_backtest import BacktestTimeSeries
from fixtures.factories.domain import ContextBuilder


def _ctx(pairs: tuple[str, ...] = ("BTC/USDC", "ETH/USDC")) -> ...:
    return ContextBuilder(pairs=pairs).trending_up().build()


def _prices(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64) * 100.0


def test_empty_contexts_raises() -> None:
    with pytest.raises(ValueError, match="contexts must not be empty"):
        BacktestTimeSeries(contexts=(), next_prices=())


def test_mismatched_lengths_raises() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="contexts and next_prices length mismatch"):
        BacktestTimeSeries(
            contexts=(ctx, ctx),
            next_prices=(_prices(2),),
        )


def test_inconsistent_pair_count_raises() -> None:
    ctx_2 = _ctx(("BTC/USDC", "ETH/USDC"))
    ctx_3 = _ctx(("BTC/USDC", "ETH/USDC", "SOL/USDC"))
    with pytest.raises(ValueError, match="bar 1 has 3 pairs, expected 2"):
        BacktestTimeSeries(
            contexts=(ctx_2, ctx_3),
            next_prices=(_prices(2), _prices(3)),
        )


def test_valid_construction() -> None:
    ctx = _ctx()
    series = BacktestTimeSeries(
        contexts=(ctx, ctx, ctx),
        next_prices=(_prices(2), _prices(2), _prices(2)),
    )
    assert series.n_bars == 3
    assert series.n_pairs == 2

"""Validation tests for BacktestTimeSeries."""

import numpy as np
import pytest

from dojiwick.application.use_cases.run_backtest import BacktestTimeSeries
from fixtures.factories.domain import ContextBuilder


def _ctx(pairs: tuple[str, ...] = ("BTC/USDC", "ETH/USDC")) -> ...:
    return ContextBuilder(pairs=pairs).trending_up().build()


def _prices(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64) * 100.0


def _mask(n_bars: int, n_pairs: int) -> np.ndarray:
    return np.ones((n_bars, n_pairs), dtype=np.bool_)


def test_empty_contexts_raises() -> None:
    with pytest.raises(ValueError, match="contexts must not be empty"):
        BacktestTimeSeries(contexts=(), next_prices=(), active_mask=np.ones((0, 0), dtype=np.bool_))


def test_mismatched_lengths_raises() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="contexts and next_prices length mismatch"):
        BacktestTimeSeries(
            contexts=(ctx, ctx),
            next_prices=(_prices(2),),
            active_mask=_mask(2, 2),
        )


def test_inconsistent_pair_count_raises() -> None:
    ctx_2 = _ctx(("BTC/USDC", "ETH/USDC"))
    ctx_3 = _ctx(("BTC/USDC", "ETH/USDC", "SOL/USDC"))
    with pytest.raises(ValueError, match="bar 1 has 3 pairs, expected 2"):
        BacktestTimeSeries(
            contexts=(ctx_2, ctx_3),
            next_prices=(_prices(2), _prices(3)),
            active_mask=_mask(2, 2),
        )


def test_valid_construction() -> None:
    ctx = _ctx()
    series = BacktestTimeSeries(
        contexts=(ctx, ctx, ctx),
        next_prices=(_prices(2), _prices(2), _prices(2)),
        active_mask=_mask(3, 2),
    )
    assert series.n_bars == 3
    assert series.n_pairs == 2


def test_active_mask_shape_mismatch_raises() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="active_mask shape"):
        BacktestTimeSeries(
            contexts=(ctx, ctx),
            next_prices=(_prices(2), _prices(2)),
            active_mask=_mask(3, 2),  # wrong n_bars
        )


def test_active_mask_dtype_mismatch_raises() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="active_mask dtype must be bool"):
        BacktestTimeSeries(
            contexts=(ctx, ctx),
            next_prices=(_prices(2), _prices(2)),
            active_mask=np.ones((2, 2), dtype=np.float64),
        )

"""Tests for portfolio state evolution between bars."""

from datetime import UTC, datetime

import numpy as np

from dojiwick.compute.kernels.pnl.portfolio_evolution import compute_bar_net_pnl, evolve_portfolio
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.models.value_objects.batch_models import (
    BatchExecutionIntent,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.models.value_objects.cost_model import CostModel

_ZERO_COST = CostModel(slippage_bps=0.0, fee_bps=0.0, fee_multiplier=2.0)


def _make_portfolio(equity: float, day_start: float | None = None) -> BatchPortfolioSnapshot:
    size = 1
    return BatchPortfolioSnapshot(
        equity_usd=np.array([equity], dtype=np.float64),
        day_start_equity_usd=np.array([day_start or equity], dtype=np.float64),
        open_positions_total=np.zeros(size, dtype=np.int64),
        has_open_position=np.zeros(size, dtype=np.bool_),
        unrealized_pnl_usd=np.zeros(size, dtype=np.float64),
    )


def _make_intents(action: int, entry_price: float, quantity: float, notional: float) -> BatchExecutionIntent:
    return BatchExecutionIntent(
        pairs=("BTC/USDC",),
        action=np.array([action], dtype=np.int64),
        quantity=np.array([quantity], dtype=np.float64),
        notional_usd=np.array([notional], dtype=np.float64),
        entry_price=np.array([entry_price], dtype=np.float64),
        stop_price=np.array([entry_price * 0.95], dtype=np.float64),
        take_profit_price=np.array([entry_price * 1.10], dtype=np.float64),
        strategy_name=("test",),
        strategy_variant=("v1",),
        active_mask=np.array([True], dtype=np.bool_),
    )


def test_compute_bar_net_pnl_long_profit() -> None:
    intents = _make_intents(TradeAction.BUY.value, 100.0, 1.0, 100.0)
    next_prices = np.array([110.0], dtype=np.float64)
    pnl = compute_bar_net_pnl(intents, next_prices, _ZERO_COST)
    assert pnl[0] > 0.0
    np.testing.assert_allclose(pnl[0], 10.0, atol=0.01)


def test_compute_bar_net_pnl_short_profit() -> None:
    intents = _make_intents(TradeAction.SHORT.value, 100.0, 1.0, 100.0)
    next_prices = np.array([90.0], dtype=np.float64)
    pnl = compute_bar_net_pnl(intents, next_prices, _ZERO_COST)
    assert pnl[0] > 0.0
    np.testing.assert_allclose(pnl[0], 10.0, atol=0.01)


def test_evolve_portfolio_adds_pnl() -> None:
    portfolio = _make_portfolio(10_000.0)
    bar_pnl = np.array([500.0], dtype=np.float64)
    t1 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    t0 = datetime(2025, 1, 1, 11, 0, tzinfo=UTC)
    result = evolve_portfolio(portfolio, bar_pnl, t1, t0)
    np.testing.assert_allclose(result.equity_usd, [10_500.0])


def test_evolve_portfolio_floors_at_zero() -> None:
    portfolio = _make_portfolio(100.0)
    bar_pnl = np.array([-500.0], dtype=np.float64)
    t1 = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    result = evolve_portfolio(portfolio, bar_pnl, t1, None)
    np.testing.assert_allclose(result.equity_usd, [0.0])


def test_evolve_portfolio_resets_day_start_on_day_boundary() -> None:
    portfolio = _make_portfolio(10_000.0, day_start=9_000.0)
    bar_pnl = np.array([200.0], dtype=np.float64)
    prev = datetime(2025, 1, 1, 23, 0, tzinfo=UTC)
    curr = datetime(2025, 1, 2, 0, 0, tzinfo=UTC)
    result = evolve_portfolio(portfolio, bar_pnl, curr, prev)
    np.testing.assert_allclose(result.equity_usd, [10_200.0])
    np.testing.assert_allclose(result.day_start_equity_usd, [10_200.0])


def test_evolve_portfolio_keeps_day_start_within_same_day() -> None:
    portfolio = _make_portfolio(10_000.0, day_start=9_000.0)
    bar_pnl = np.array([200.0], dtype=np.float64)
    prev = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    curr = datetime(2025, 1, 1, 11, 0, tzinfo=UTC)
    result = evolve_portfolio(portfolio, bar_pnl, curr, prev)
    np.testing.assert_allclose(result.equity_usd, [10_200.0])
    np.testing.assert_allclose(result.day_start_equity_usd, [9_000.0])

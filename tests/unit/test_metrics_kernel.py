"""Metrics kernel tests."""

import numpy as np

from dojiwick.compute.kernels.metrics.summarize import summarize


def test_zero_trades_returns_empty_summary() -> None:
    net_pnl = np.array([0.0, 0.0], dtype=np.float64)
    notional = np.array([0.0, 0.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.trades == 0
    assert result.total_pnl_usd == 0.0
    assert result.win_rate == 0.0
    assert result.sharpe_like == 0.0
    assert result.max_drawdown_pct == 0.0


def test_known_pnl_vector() -> None:
    net_pnl = np.array([10.0, -5.0, 20.0, -3.0, 15.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.trades == 5
    assert result.total_pnl_usd == 37.0
    assert result.win_rate == 3 / 5
    assert abs(result.expectancy_usd - 7.4) < 0.01


def test_sharpe_nonzero_for_varied_returns() -> None:
    net_pnl = np.array([10.0, -5.0, 20.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.sharpe_like != 0.0


def test_drawdown_computed() -> None:
    net_pnl = np.array([10.0, -20.0, 5.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.max_drawdown_pct > 0.0


def test_all_winners() -> None:
    net_pnl = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.win_rate == 1.0
    assert result.max_drawdown_pct == 0.0


def test_annualization_scales_sharpe_sortino() -> None:
    net_pnl = np.array([10.0, -5.0, 20.0, -3.0, 15.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)

    base = summarize(net_pnl, notional, n_bars=0).summary
    ann = summarize(net_pnl, notional, n_bars=1000, bars_per_year=8760.0).summary

    # Annualized values should be larger in magnitude than per-trade
    assert abs(ann.sharpe_like) > abs(base.sharpe_like)
    assert abs(ann.sortino) > abs(base.sortino)


def test_avg_notional_usd_computed() -> None:
    net_pnl = np.array([10.0, -5.0], dtype=np.float64)
    notional = np.array([100.0, 200.0], dtype=np.float64)

    result = summarize(net_pnl, notional).summary
    assert result.avg_notional_usd == 150.0


def test_trade_returns_always_returned() -> None:
    net_pnl = np.array([10.0, -5.0, 20.0], dtype=np.float64)
    notional = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    result = summarize(net_pnl, notional)
    assert len(result.trade_returns) == 3

    # Zero-trade case returns empty array
    empty_result = summarize(np.zeros(2), np.zeros(2))
    assert len(empty_result.trade_returns) == 0

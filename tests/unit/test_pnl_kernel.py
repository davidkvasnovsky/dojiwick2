"""PnL kernel unit tests."""

import numpy as np
from numpy.testing import assert_allclose

from dojiwick.domain.enums import TradeAction
from dojiwick.compute.kernels.pnl.pnl import apply_slippage, gross_pnl, net_pnl


def test_slippage_worsens_buy_entry() -> None:
    entry = np.array([100.0], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    adjusted = apply_slippage(entry, action, slippage_bps=10.0)
    assert adjusted[0] > entry[0]


def test_slippage_improves_short_entry() -> None:
    entry = np.array([100.0], dtype=np.float64)
    action = np.array([TradeAction.SHORT.value], dtype=np.int64)
    adjusted = apply_slippage(entry, action, slippage_bps=10.0)
    assert adjusted[0] < entry[0]


def test_gross_pnl_long_profit() -> None:
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    entry = np.array([100.0], dtype=np.float64)
    exit_price = np.array([110.0], dtype=np.float64)
    qty = np.array([1.0], dtype=np.float64)
    result = gross_pnl(action, entry, exit_price, qty)
    assert_allclose(result.item(0), 10.0)


def test_gross_pnl_short_profit() -> None:
    action = np.array([TradeAction.SHORT.value], dtype=np.int64)
    entry = np.array([100.0], dtype=np.float64)
    exit_price = np.array([90.0], dtype=np.float64)
    qty = np.array([1.0], dtype=np.float64)
    result = gross_pnl(action, entry, exit_price, qty)
    assert_allclose(result.item(0), 10.0)


def test_net_pnl_deducts_fees() -> None:
    gross = np.array([10.0], dtype=np.float64)
    notional = np.array([100.0], dtype=np.float64)
    result = net_pnl(gross, notional, fee_bps=4.0)
    expected_fee = 100.0 * (4.0 / 10_000) * 2
    assert_allclose(result.item(0), 10.0 - expected_fee)


def test_zero_quantity_returns_zero_pnl() -> None:
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    entry = np.array([100.0], dtype=np.float64)
    exit_price = np.array([110.0], dtype=np.float64)
    qty = np.array([0.0], dtype=np.float64)
    result = gross_pnl(action, entry, exit_price, qty)
    assert result[0] == 0.0

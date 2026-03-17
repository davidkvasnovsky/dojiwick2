"""Tests for liquidation modeling."""

import numpy as np

from dojiwick.compute.kernels.pnl.liquidation import cap_pnl_at_margin, check_liquidation
from dojiwick.domain.enums import TradeAction


def test_no_liquidation_at_1x_leverage() -> None:
    entry = np.array([100.0], dtype=np.float64)
    next_price = np.array([0.01], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    result = check_liquidation(entry, next_price, leverage=1.0, action=action)
    assert not result[0]


def test_long_liquidation_at_10x() -> None:
    entry = np.array([100.0], dtype=np.float64)
    # Liq price = 100 * (1 - 1/10) = 90
    next_price = np.array([89.0], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    result = check_liquidation(entry, next_price, leverage=10.0, action=action)
    assert result[0]


def test_long_not_liquidated_above_threshold() -> None:
    entry = np.array([100.0], dtype=np.float64)
    next_price = np.array([91.0], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    result = check_liquidation(entry, next_price, leverage=10.0, action=action)
    assert not result[0]


def test_short_liquidation_at_5x() -> None:
    entry = np.array([100.0], dtype=np.float64)
    # Liq price = 100 * (1 + 1/5) = 120
    next_price = np.array([121.0], dtype=np.float64)
    action = np.array([TradeAction.SHORT.value], dtype=np.int64)
    result = check_liquidation(entry, next_price, leverage=5.0, action=action)
    assert result[0]


def test_cap_pnl_at_margin() -> None:
    pnl = np.array([-200.0], dtype=np.float64)
    notional = np.array([1000.0], dtype=np.float64)
    liquidated = np.array([True], dtype=np.bool_)
    result = cap_pnl_at_margin(pnl, notional, leverage=10.0, liquidated=liquidated)
    # Margin = 1000 / 10 = 100, so loss capped at -100
    np.testing.assert_allclose(result, [-100.0])


def test_cap_pnl_no_change_when_not_liquidated() -> None:
    pnl = np.array([-50.0], dtype=np.float64)
    notional = np.array([1000.0], dtype=np.float64)
    liquidated = np.array([False], dtype=np.bool_)
    result = cap_pnl_at_margin(pnl, notional, leverage=10.0, liquidated=liquidated)
    np.testing.assert_allclose(result, [-50.0])


# --- Maintenance margin tests ---


def test_liquidation_with_maintenance_margin() -> None:
    """With MMR, liquidation price is closer to entry than legacy."""
    entry = np.array([100.0], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    # 10x leverage, MMR=0.005: liq = 100 * (1 - (0.1 - 0.005)) = 90.5
    # Legacy liq = 100 * (1 - 0.1) = 90.0
    # MMR liq is CLOSER to entry (90.5 > 90.0) — liquidation triggers sooner

    # Price at 90.25: NOT liquidated under legacy (90.25 > 90.0), but IS with MMR (90.25 < 90.5)
    next_price = np.array([90.25], dtype=np.float64)
    result_legacy = check_liquidation(entry, next_price, leverage=10.0, action=action)
    result_mmr = check_liquidation(entry, next_price, leverage=10.0, action=action, maintenance_margin_rate=0.005)
    assert not result_legacy[0], "90.25 > 90.0 legacy liq: not liquidated"
    assert result_mmr[0], "90.25 < 90.5 MMR liq: liquidated"

    # Price at 91.0: not liquidated under either model
    next_price_safe = np.array([91.0], dtype=np.float64)
    result_safe = check_liquidation(entry, next_price_safe, leverage=10.0, action=action, maintenance_margin_rate=0.005)
    assert not result_safe[0], "91.0 > 90.5 MMR liq: not liquidated"


def test_liquidation_legacy_fallback() -> None:
    """maintenance_margin_rate=0 produces same results as legacy formula."""
    entry = np.array([100.0], dtype=np.float64)
    action = np.array([TradeAction.BUY.value], dtype=np.int64)
    next_price = np.array([89.0], dtype=np.float64)

    result_legacy = check_liquidation(entry, next_price, leverage=10.0, action=action)
    result_zero_mmr = check_liquidation(entry, next_price, leverage=10.0, action=action, maintenance_margin_rate=0.0)
    assert result_legacy[0] == result_zero_mmr[0]


def test_short_liquidation_with_maintenance_margin() -> None:
    """Short positions: MMR moves liq price closer to entry (lower value)."""
    entry = np.array([100.0], dtype=np.float64)
    action = np.array([TradeAction.SHORT.value], dtype=np.int64)
    # 10x leverage, MMR=0.005: liq = 100 * (1 + (0.1 - 0.005)) = 109.5
    # Legacy liq = 100 * (1 + 0.1) = 110.0
    # MMR liq is CLOSER to entry (109.5 < 110.0) — liquidation triggers sooner

    # Price at 109.75: NOT liquidated under legacy (109.75 < 110.0), but IS with MMR (109.75 >= 109.5)
    next_price = np.array([109.75], dtype=np.float64)
    result_legacy = check_liquidation(entry, next_price, leverage=10.0, action=action)
    result_mmr = check_liquidation(entry, next_price, leverage=10.0, action=action, maintenance_margin_rate=0.005)
    assert not result_legacy[0], "109.75 < 110.0 legacy liq: not liquidated"
    assert result_mmr[0], "109.75 >= 109.5 MMR liq: liquidated"

    # Price at 109.0: not liquidated under either model
    next_price_safe = np.array([109.0], dtype=np.float64)
    result_safe = check_liquidation(entry, next_price_safe, leverage=10.0, action=action, maintenance_margin_rate=0.005)
    assert not result_safe[0], "109.0 < 109.5 MMR liq: not liquidated"

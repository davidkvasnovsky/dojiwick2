"""Tests for partial fill simulation kernel."""

from __future__ import annotations

import numpy as np

from dojiwick.compute.kernels.pnl.partial_fill import apply_fill_ratio, compute_fill_ratio
from dojiwick.domain.enums import TradeAction


def test_small_order_gets_full_fill() -> None:
    """Orders well below threshold get ratio 1.0."""
    ratio = compute_fill_ratio(
        notional_usd=np.array([10.0]),
        bar_volume=np.array([10000.0]),
        entry_price=np.array([100.0]),
        action=np.array([TradeAction.BUY]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    assert ratio[0] == 1.0


def test_large_order_gets_partial_fill() -> None:
    """Order exceeding threshold gets partial fill."""
    # notional = 100, bar_volume_usd = 1000 * 1.0 = 1000
    # raw_ratio = 0.05 * 1000 / 100 = 0.5
    ratio = compute_fill_ratio(
        notional_usd=np.array([100.0]),
        bar_volume=np.array([1000.0]),
        entry_price=np.array([1.0]),
        action=np.array([TradeAction.BUY]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    np.testing.assert_allclose(ratio, [0.5])


def test_min_ratio_clamp() -> None:
    """Very large orders are clamped to min_ratio."""
    # notional = 10000, bar_volume_usd = 10 * 1.0 = 10
    # raw_ratio = 0.05 * 10 / 10000 = 0.00005 → clamped to 0.1
    ratio = compute_fill_ratio(
        notional_usd=np.array([10000.0]),
        bar_volume=np.array([10.0]),
        entry_price=np.array([1.0]),
        action=np.array([TradeAction.BUY]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    np.testing.assert_allclose(ratio, [0.1])


def test_hold_always_full_fill() -> None:
    """HOLD actions always get ratio 1.0 regardless of volume."""
    ratio = compute_fill_ratio(
        notional_usd=np.array([10000.0]),
        bar_volume=np.array([1.0]),
        entry_price=np.array([1.0]),
        action=np.array([TradeAction.HOLD]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    assert ratio[0] == 1.0


def test_zero_notional_no_crash() -> None:
    """Zero notional doesn't cause division by zero."""
    ratio = compute_fill_ratio(
        notional_usd=np.array([0.0]),
        bar_volume=np.array([1000.0]),
        entry_price=np.array([100.0]),
        action=np.array([TradeAction.BUY]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    # safe_notional = 1.0, raw = 0.05 * 100000 / 1.0 = 5000, clamped to 1.0
    assert ratio[0] == 1.0


def test_apply_fill_ratio_scales() -> None:
    """apply_fill_ratio scales quantity and notional."""
    qty, notional = apply_fill_ratio(
        quantity=np.array([10.0, 20.0]),
        notional_usd=np.array([100.0, 200.0]),
        fill_ratio=np.array([0.5, 1.0]),
    )
    np.testing.assert_allclose(qty, [5.0, 20.0])
    np.testing.assert_allclose(notional, [50.0, 200.0])


def test_multi_pair_mixed_actions() -> None:
    """Mixed BUY/SHORT/HOLD across pairs."""
    ratio = compute_fill_ratio(
        notional_usd=np.array([100.0, 100.0, 100.0]),
        bar_volume=np.array([1000.0, 1000.0, 1000.0]),
        entry_price=np.array([1.0, 1.0, 1.0]),
        action=np.array([TradeAction.BUY, TradeAction.SHORT, TradeAction.HOLD]),
        threshold_pct=0.05,
        min_ratio=0.1,
    )
    np.testing.assert_allclose(ratio[0], 0.5)
    np.testing.assert_allclose(ratio[1], 0.5)
    assert ratio[2] == 1.0

"""Tests for entry price resolution kernel."""

from __future__ import annotations

import numpy as np

from dojiwick.compute.kernels.pnl.entry_price import resolve_entry_price
from dojiwick.domain.enums import EntryPriceModel, TradeAction


def _make_arrays() -> dict[str, np.ndarray]:
    return {
        "close": np.array([100.0, 200.0, 300.0]),
        "next_open": np.array([101.0, 201.0, 301.0]),
        "next_high": np.array([110.0, 210.0, 310.0]),
        "next_low": np.array([90.0, 190.0, 290.0]),
        "next_close": np.array([105.0, 205.0, 305.0]),
        "action": np.array([TradeAction.BUY, TradeAction.SHORT, TradeAction.HOLD]),
    }


def test_close_model() -> None:
    d = _make_arrays()
    result = resolve_entry_price(EntryPriceModel.CLOSE, **d)
    np.testing.assert_array_equal(result, d["close"])


def test_next_open_model() -> None:
    d = _make_arrays()
    result = resolve_entry_price(EntryPriceModel.NEXT_OPEN, **d)
    np.testing.assert_array_equal(result, d["next_open"])


def test_vwap_proxy_model() -> None:
    d = _make_arrays()
    result = resolve_entry_price(EntryPriceModel.VWAP_PROXY, **d)
    expected = (d["next_open"] + d["next_high"] + d["next_low"] + d["next_close"]) / 4.0
    np.testing.assert_allclose(result, expected)


def test_worst_case_model() -> None:
    d = _make_arrays()
    result = resolve_entry_price(EntryPriceModel.WORST_CASE, **d)
    # BUY -> next_high (worst entry for longs)
    assert result[0] == 110.0
    # SHORT -> next_low (worst entry for shorts)
    assert result[1] == 190.0
    # HOLD -> close (unchanged)
    assert result[2] == 300.0


def test_close_model_returns_copy() -> None:
    """Close model returns a copy, not the original array."""
    d = _make_arrays()
    result = resolve_entry_price(EntryPriceModel.CLOSE, **d)
    result[0] = 999.0
    assert d["close"][0] == 100.0


def test_all_holds_worst_case() -> None:
    """Worst case with all HOLD returns close prices."""
    d = _make_arrays()
    d["action"] = np.array([TradeAction.HOLD, TradeAction.HOLD, TradeAction.HOLD])
    result = resolve_entry_price(EntryPriceModel.WORST_CASE, **d)
    np.testing.assert_array_equal(result, d["close"])

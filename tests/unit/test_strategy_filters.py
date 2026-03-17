"""Tests for shared strategy filter predicates in _filters.py."""

import numpy as np
import pytest

from dojiwick.compute.kernels.strategy._filters import (
    ema_triple_aligned_down,
    ema_triple_aligned_up,
    macd_direction_aligned,
)


@pytest.mark.unit
class TestEmaTripleAligned:
    def test_aligned_up(self) -> None:
        fast = np.array([30.0, 20.0, 10.0])
        slow = np.array([20.0, 25.0, 5.0])
        base = np.array([10.0, 30.0, 15.0])
        result = ema_triple_aligned_up(fast, slow, base)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_aligned_down(self) -> None:
        fast = np.array([10.0, 20.0, 30.0])
        slow = np.array([20.0, 15.0, 25.0])
        base = np.array([30.0, 10.0, 20.0])
        result = ema_triple_aligned_down(fast, slow, base)
        np.testing.assert_array_equal(result, [True, False, False])


@pytest.mark.unit
class TestMacdDirectionAligned:
    def test_buy_positive_macd(self) -> None:
        macd = np.array([1.0, -1.0, 0.5])
        buy = np.array([True, True, False])
        short = np.array([False, False, False])
        result = macd_direction_aligned(macd, buy, short)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_short_negative_macd(self) -> None:
        macd = np.array([-1.0, 1.0, -0.5])
        buy = np.array([False, False, False])
        short = np.array([True, True, False])
        result = macd_direction_aligned(macd, buy, short)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_hold_always_false(self) -> None:
        macd = np.array([1.0, -1.0, 0.0, 5.0])
        buy = np.array([False, False, False, False])
        short = np.array([False, False, False, False])
        result = macd_direction_aligned(macd, buy, short)
        np.testing.assert_array_equal(result, [False, False, False, False])

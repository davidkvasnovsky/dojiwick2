"""Unit tests for the indicator compute kernel."""

import numpy as np

from dojiwick.compute.kernels.indicators.compute import (
    adx,
    atr,
    bollinger_bands,
    compute_indicators,
    ema,
    rsi,
)
from dojiwick.domain.indicator_schema import INDICATOR_COUNT, INDICATOR_INDEX


class TestEma:
    def test_constant_input_converges(self) -> None:
        values = np.full(100, 42.0, dtype=np.float64)
        result = ema(values, 12)
        np.testing.assert_allclose(result[11], 42.0, atol=1e-10)
        np.testing.assert_allclose(result[-1], 42.0, atol=1e-10)

    def test_warmup_is_nan(self) -> None:
        values = np.arange(1, 51, dtype=np.float64)
        result = ema(values, 20)
        assert np.all(np.isnan(result[:19]))
        assert np.isfinite(result[19])

    def test_too_short_all_nan(self) -> None:
        values = np.array([1.0, 2.0], dtype=np.float64)
        result = ema(values, 5)
        assert np.all(np.isnan(result))


class TestRsi:
    def test_all_gains_near_100(self) -> None:
        close = np.arange(1.0, 100.0, dtype=np.float64)
        result = rsi(close, 14)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert finite[-1] > 95.0

    def test_all_losses_near_0(self) -> None:
        close = np.arange(100.0, 1.0, -1.0, dtype=np.float64)
        result = rsi(close, 14)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert finite[-1] < 5.0

    def test_warmup_is_nan(self) -> None:
        close = np.random.default_rng(42).uniform(90, 110, 100).astype(np.float64)
        result = rsi(close, 14)
        assert np.all(np.isnan(result[:14]))


class TestAtr:
    def test_constant_range_stable(self) -> None:
        n = 100
        high = np.full(n, 105.0, dtype=np.float64)
        low = np.full(n, 95.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)
        result = atr(high, low, close, 14)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        np.testing.assert_allclose(finite[-1], 10.0, rtol=0.01)


class TestAdx:
    def test_strong_trend_high_value(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.uniform(0.5, 1.5, n))
        high = base + 2.0
        low = base - 0.5
        close = base + 0.5
        result = adx(high, low, close, 14)
        finite = result[np.isfinite(result)]
        assert len(finite) > 0
        assert finite[-1] > 25.0


class TestBollingerBands:
    def test_constant_close_symmetric(self) -> None:
        close = np.full(50, 100.0, dtype=np.float64)
        upper, lower = bollinger_bands(close, 20, 2.0)
        finite_upper = upper[np.isfinite(upper)]
        finite_lower = lower[np.isfinite(lower)]
        assert len(finite_upper) > 0
        np.testing.assert_allclose(finite_upper, 100.0, atol=1e-10)
        np.testing.assert_allclose(finite_lower, 100.0, atol=1e-10)

    def test_warmup_is_nan(self) -> None:
        close = np.random.default_rng(42).uniform(90, 110, 50).astype(np.float64)
        upper, lower = bollinger_bands(close, 20)
        assert np.all(np.isnan(upper[:19]))
        assert np.all(np.isnan(lower[:19]))

    def test_non_trivial_values(self) -> None:
        close = np.array([10.0, 12.0, 11.0, 13.0, 15.0], dtype=np.float64)
        upper, lower = bollinger_bands(close, period=3, num_std=2.0)
        assert np.all(np.isnan(upper[:2]))
        assert np.all(np.isnan(lower[:2]))
        np.testing.assert_allclose(upper[2], 12.632993, rtol=1e-5)
        np.testing.assert_allclose(lower[2], 9.367007, rtol=1e-5)
        np.testing.assert_allclose(upper[3], 13.632993, rtol=1e-5)
        np.testing.assert_allclose(lower[3], 10.367007, rtol=1e-5)
        np.testing.assert_allclose(upper[4], 16.265986, rtol=1e-5)
        np.testing.assert_allclose(lower[4], 9.734014, rtol=1e-5)


def _make_price_series(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (close, high, low) arrays for indicator tests."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.abs(close) + 10.0
    high = close + rng.uniform(1, 3, n)
    low = close - rng.uniform(1, 3, n)
    return close, high, low


class TestComputeIndicators:
    def test_shape(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        close = rng.uniform(90, 110, n).astype(np.float64)
        high = close + rng.uniform(1, 5, n)
        low = close - rng.uniform(1, 5, n)
        result = compute_indicators(close, high, low)
        assert result.shape == (n, INDICATOR_COUNT)
        assert result.dtype == np.float64

    def test_warmup_bars_are_nan(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        close = rng.uniform(90, 110, n).astype(np.float64)
        high = close + rng.uniform(1, 5, n)
        low = close - rng.uniform(1, 5, n)
        result = compute_indicators(close, high, low)
        # First 50 bars (ema_base_period) should have NaN in ema_base column at minimum
        assert np.any(np.isnan(result[:49]))

    def test_post_warmup_finite(self) -> None:
        close, high, low = _make_price_series()
        volume = np.random.default_rng(42).uniform(100, 1000, len(close)).astype(np.float64)
        result = compute_indicators(close, high, low, volume=volume)
        # After bar 210 (generous warmup for ema_trend=200), all values should be finite
        post_warmup = result[210:]
        assert np.all(np.isfinite(post_warmup)), (
            f"Non-finite values found at rows: {np.argwhere(~np.isfinite(post_warmup))}"
        )

    def test_bb_mid_equals_band_average(self) -> None:
        close, high, low = _make_price_series()
        result = compute_indicators(close, high, low)
        bb_upper = result[:, INDICATOR_INDEX["bb_upper"]]
        bb_lower = result[:, INDICATOR_INDEX["bb_lower"]]
        bb_mid = result[:, INDICATOR_INDEX["bb_mid"]]
        expected = (bb_upper + bb_lower) / 2.0
        finite = np.isfinite(bb_mid)
        np.testing.assert_allclose(bb_mid[finite], expected[finite], atol=1e-10)

    def test_macd_histogram_equals_line_minus_signal(self) -> None:
        close, high, low = _make_price_series()
        result = compute_indicators(close, high, low)
        ema_fast = result[:, INDICATOR_INDEX["ema_fast"]]
        ema_slow = result[:, INDICATOR_INDEX["ema_slow"]]
        macd_line = ema_fast - ema_slow
        macd_signal = result[:, INDICATOR_INDEX["macd_signal"]]
        macd_hist = result[:, INDICATOR_INDEX["macd_histogram"]]
        finite = np.isfinite(macd_hist)
        np.testing.assert_allclose(macd_hist[finite], (macd_line - macd_signal)[finite], atol=1e-10)

    def test_macd_post_warmup_finite(self) -> None:
        close, high, low = _make_price_series()
        result = compute_indicators(close, high, low)
        # EMA slow (26) + signal EMA (9) = 34 warmup bars; use 40 for safety
        macd_hist = result[40:, INDICATOR_INDEX["macd_histogram"]]
        macd_sig = result[40:, INDICATOR_INDEX["macd_signal"]]
        assert np.all(np.isfinite(macd_hist))
        assert np.all(np.isfinite(macd_sig))

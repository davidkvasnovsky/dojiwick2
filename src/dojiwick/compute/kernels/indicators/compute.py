"""Pure numpy indicator computation kernel.

Computes all 13 indicators from ``INDICATOR_NAMES`` using hand-rolled
numpy implementations. Column ordering follows ``INDICATOR_INDEX``
from ``domain/indicator_schema.py``.

Warmup bars are filled with ``np.nan`` (not 0.0) so that the regime
classifier's ``valid_mask`` rejects them via ``np.isfinite()``.

``ema``, ``rsi``, ``atr``, and ``adx`` use sequential Python loops
because Wilder's smoothing carries state forward bar-by-bar and cannot
be vectorized.
"""

import numpy as np

from dojiwick.domain.indicator_schema import INDICATOR_COUNT, INDICATOR_INDEX
from dojiwick.domain.type_aliases import FloatMatrix, FloatVector


def ema(values: FloatVector, period: int) -> FloatVector:
    """Exponential moving average with SMA seed for the first *period* bars."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    out[period - 1] = np.mean(values[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(close: FloatVector, period: int = 14) -> FloatVector:
    """Wilder's smoothed RSI."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return out


def atr(high: FloatVector, low: FloatVector, close: FloatVector, period: int = 14) -> FloatVector:
    """Average True Range using Wilder's smoothing."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))

    out[period] = np.mean(tr[1 : period + 1])
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period

    return out


def adx(
    high: FloatVector,
    low: FloatVector,
    close: FloatVector,
    period: int = 14,
    *,
    precomputed_atr: FloatVector | None = None,
) -> FloatVector:
    """Average Directional Index (+DM/-DM -> +DI/-DI -> DX -> smooth)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * period + 1:
        return out

    up_move = np.diff(high)
    down_move = -np.diff(low)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = precomputed_atr if precomputed_atr is not None else atr(high, low, close, period)

    # Wilder smooth +DM and -DM
    smooth_plus = np.full(n, np.nan, dtype=np.float64)
    smooth_minus = np.full(n, np.nan, dtype=np.float64)
    smooth_plus[period] = np.sum(plus_dm[:period])
    smooth_minus[period] = np.sum(minus_dm[:period])
    for i in range(period + 1, n):
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i - 1]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i - 1]

    atr_safe = np.where(atr_vals > 0, atr_vals, 1.0)
    plus_di = 100.0 * smooth_plus / atr_safe
    minus_di = 100.0 * smooth_minus / atr_safe

    di_sum = plus_di + minus_di
    di_sum_safe = np.where(di_sum > 0, di_sum, 1.0)
    dx = 100.0 * np.abs(plus_di - minus_di) / di_sum_safe

    # Smooth DX to get ADX
    adx_start = 2 * period
    if adx_start < n:
        out[adx_start] = np.nanmean(dx[period + 1 : adx_start + 1])
        for i in range(adx_start + 1, n):
            if np.isnan(out[i - 1]) or np.isnan(dx[i]):
                continue
            out[i] = (out[i - 1] * (period - 1) + dx[i]) / period

    return out


def bollinger_bands(close: FloatVector, period: int = 20, num_std: float = 2.0) -> tuple[FloatVector, FloatVector]:
    """Rolling SMA +/- num_std * rolling std -> (upper, lower)."""
    n = len(close)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return upper, lower

    cumsum = np.cumsum(close)
    cumsum = np.concatenate(([0.0], cumsum))
    rolling_sum = cumsum[period:] - cumsum[:-period]
    sma = rolling_sum / period

    cumsum_sq = np.cumsum(close**2)
    cumsum_sq = np.concatenate(([0.0], cumsum_sq))
    rolling_sum_sq = cumsum_sq[period:] - cumsum_sq[:-period]
    variance = np.maximum(rolling_sum_sq / period - sma**2, 0.0)
    std = np.sqrt(variance)

    upper[period - 1 :] = sma + num_std * std
    lower[period - 1 :] = sma - num_std * std

    return upper, lower


def compute_indicators(
    close: FloatVector,
    high: FloatVector,
    low: FloatVector,
    *,
    volume: FloatVector | None = None,
    # Industry-standard TA defaults; keyword-only for testability.
    rsi_period: int = 14,
    ema_fast_period: int = 12,
    ema_slow_period: int = 26,
    ema_base_period: int = 50,
    ema_trend_period: int = 200,
    atr_period: int = 14,
    adx_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    volume_ema_period: int = 20,
) -> FloatMatrix:
    """Compute all 13 indicators and return an ``(n_bars, 13)`` matrix.

    Warmup bars contain ``np.nan`` — driven by ``ema_trend_period`` (200 bars).
    """
    n = len(close)
    out = np.full((n, INDICATOR_COUNT), np.nan, dtype=np.float64)

    atr_vals = atr(high, low, close, atr_period)
    out[:, INDICATOR_INDEX["rsi"]] = rsi(close, rsi_period)
    out[:, INDICATOR_INDEX["adx"]] = adx(high, low, close, adx_period, precomputed_atr=atr_vals)
    out[:, INDICATOR_INDEX["atr"]] = atr_vals
    out[:, INDICATOR_INDEX["ema_fast"]] = ema(close, ema_fast_period)
    out[:, INDICATOR_INDEX["ema_slow"]] = ema(close, ema_slow_period)
    out[:, INDICATOR_INDEX["ema_base"]] = ema(close, ema_base_period)
    bb_upper, bb_lower = bollinger_bands(close, bb_period, bb_std)
    out[:, INDICATOR_INDEX["bb_upper"]] = bb_upper
    out[:, INDICATOR_INDEX["bb_lower"]] = bb_lower
    out[:, INDICATOR_INDEX["bb_mid"]] = (bb_upper + bb_lower) / 2.0

    out[:, INDICATOR_INDEX["ema_trend"]] = ema(close, ema_trend_period)

    macd_line = out[:, INDICATOR_INDEX["ema_fast"]] - out[:, INDICATOR_INDEX["ema_slow"]]
    # MACD line is NaN until ema_slow warmup ends; compute signal EMA on valid suffix only
    valid_start = ema_slow_period - 1
    if n > valid_start:
        out[valid_start:, INDICATOR_INDEX["macd_signal"]] = ema(macd_line[valid_start:], 9)
    out[:, INDICATOR_INDEX["macd_histogram"]] = macd_line - out[:, INDICATOR_INDEX["macd_signal"]]

    if volume is not None:
        vol_ema = ema(volume, volume_ema_period)
        safe_vol_ema = np.where(vol_ema > 0, vol_ema, np.nan)
        out[:, INDICATOR_INDEX["volume_ema_ratio"]] = volume / safe_vol_ema

    return out

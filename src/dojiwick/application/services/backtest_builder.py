"""Build BacktestTimeSeries from historical candle data."""

import logging
from datetime import UTC, datetime

import numpy as np

from dojiwick.application.use_cases.run_backtest import BacktestTimeSeries
from dojiwick.compute.kernels.indicators.compute import compute_indicators
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.numerics import candles_to_ohlc, decimals_to_array

log = logging.getLogger(__name__)


def build_backtest_time_series(
    candles_by_pair: dict[str, tuple[Candle, ...]],
    pairs: tuple[str, ...],
    *,
    warmup_bars: int = 200,
    equity_usd: float = 10_000.0,
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
) -> BacktestTimeSeries:
    """Convert historical candles into a BacktestTimeSeries for replay.

    1. Per pair: compute indicators from OHLC.
    2. Trim first ``warmup_bars`` (NaN region).
    3. Align pairs to minimum common bar count.
    4. Build one ``BatchDecisionContext`` per bar.
    """
    pair_close: dict[str, np.ndarray] = {}
    pair_open: dict[str, np.ndarray] = {}
    pair_high: dict[str, np.ndarray] = {}
    pair_low: dict[str, np.ndarray] = {}
    pair_indicators: dict[str, np.ndarray] = {}
    pair_volume: dict[str, np.ndarray] = {}
    pair_times: dict[str, list[datetime]] = {}

    for pair in pairs:
        candles = candles_by_pair[pair]
        candles = _deduplicate_candles(candles)
        _warn_time_gaps(pair, candles)
        close, high, low = candles_to_ohlc(candles)
        open_arr = decimals_to_array([c.open for c in candles])
        volume = decimals_to_array([c.volume for c in candles])
        indicators = compute_indicators(
            close,
            high,
            low,
            volume=volume,
            rsi_period=rsi_period,
            ema_fast_period=ema_fast_period,
            ema_slow_period=ema_slow_period,
            ema_base_period=ema_base_period,
            ema_trend_period=ema_trend_period,
            atr_period=atr_period,
            adx_period=adx_period,
            bb_period=bb_period,
            bb_std=bb_std,
            volume_ema_period=volume_ema_period,
        )

        # Trim warmup
        pair_close[pair] = close[warmup_bars:]
        pair_open[pair] = open_arr[warmup_bars:]
        pair_high[pair] = high[warmup_bars:]
        pair_low[pair] = low[warmup_bars:]
        pair_indicators[pair] = indicators[warmup_bars:]
        pair_volume[pair] = volume[warmup_bars:]
        pair_times[pair] = [c.open_time for c in candles[warmup_bars:]]

    # Validate no NaN remains after warmup trim
    for pair in pairs:
        finite_mask = np.isfinite(pair_indicators[pair])
        if not np.all(finite_mask):
            nan_bars = np.where(~np.all(finite_mask, axis=1))[0]
            raise ValueError(
                f"{pair}: {len(nan_bars)} bars with NaN indicators after {warmup_bars}-bar warmup trim "
                f"(first at index {nan_bars[0]}). Increase warmup_bars or check data gaps."
            )

    _warn_pair_count_mismatch(pairs, {p: len(candles_by_pair[p]) for p in pairs})

    # Align pairs by timestamp intersection
    if len(pairs) == 1:
        # Single pair — no intersection needed
        min_bars = len(pair_close[pairs[0]])
        if min_bars < 2:
            raise ValueError(f"need at least {warmup_bars + 2} candles per pair, got too few after warmup trim")
    else:
        first_set: set[datetime] = set(pair_times[pairs[0]])
        common_set = first_set.intersection(*(set(pair_times[p]) for p in pairs[1:]))
        common_times: list[datetime] = sorted(common_set)
        if len(common_times) < 2:
            raise ValueError("insufficient common timestamps after intersection")
        max_bars = max(len(pair_times[p]) for p in pairs)
        min_coverage = 0.5
        if len(common_times) / max_bars < min_coverage:
            raise ValueError(
                f"timestamp coverage too low: {len(common_times)}/{max_bars} = "
                f"{len(common_times) / max_bars:.1%} (need {min_coverage:.0%})"
            )
        # Re-index each pair to common timestamps
        for pair in pairs:
            time_to_idx = {t: i for i, t in enumerate(pair_times[pair])}
            indices = np.array([time_to_idx[t] for t in common_times])
            pair_close[pair] = pair_close[pair][indices]
            pair_open[pair] = pair_open[pair][indices]
            pair_high[pair] = pair_high[pair][indices]
            pair_low[pair] = pair_low[pair][indices]
            pair_indicators[pair] = pair_indicators[pair][indices]
            pair_volume[pair] = pair_volume[pair][indices]
            pair_times[pair] = list(common_times)
        min_bars = len(common_times)

    n_bars = min_bars - 1  # last bar used only for next_prices
    size = len(pairs)

    # Pre-build matrices for O(1) per-bar slicing
    price_matrix = np.column_stack([pair_close[p] for p in pairs])
    open_matrix = np.column_stack([pair_open[p] for p in pairs])
    high_matrix = np.column_stack([pair_high[p] for p in pairs])
    low_matrix = np.column_stack([pair_low[p] for p in pairs])
    volume_matrix = np.column_stack([pair_volume[p] for p in pairs])
    indicator_cube = np.stack([pair_indicators[p] for p in pairs], axis=0)

    # All pairs now share the same timestamp sequence (intersection or single pair)
    times = pair_times[pairs[0]]
    if times and times[0].tzinfo is None:
        times = [t.replace(tzinfo=UTC) for t in times]

    portfolio_template = BatchPortfolioSnapshot(
        equity_usd=np.full(size, equity_usd, dtype=np.float64),
        day_start_equity_usd=np.full(size, equity_usd, dtype=np.float64),
        open_positions_total=np.zeros(size, dtype=np.int64),
        has_open_position=np.zeros(size, dtype=np.bool_),
        unrealized_pnl_usd=np.zeros(size, dtype=np.float64),
    )

    contexts: list[BatchDecisionContext] = []
    next_prices_list: list[np.ndarray] = []
    next_open_list: list[np.ndarray] = []
    next_high_list: list[np.ndarray] = []
    next_low_list: list[np.ndarray] = []

    for t in range(n_bars):
        ind_rows = indicator_cube[:, t, :]

        ctx = BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=pairs,
                observed_at=times[t],
                price=price_matrix[t],
                indicators=ind_rows,
                volume=volume_matrix[t],
            ),
            portfolio=portfolio_template,
        )
        contexts.append(ctx)
        next_prices_list.append(price_matrix[t + 1])
        next_open_list.append(open_matrix[t + 1])
        next_high_list.append(high_matrix[t + 1])
        next_low_list.append(low_matrix[t + 1])

    return BacktestTimeSeries(
        contexts=tuple(contexts),
        next_prices=tuple(next_prices_list),
        next_open=tuple(next_open_list),
        next_high=tuple(next_high_list),
        next_low=tuple(next_low_list),
    )


def _deduplicate_candles(candles: tuple[Candle, ...]) -> tuple[Candle, ...]:
    """Remove duplicate timestamps, keeping the last occurrence."""
    seen: dict[datetime, Candle] = {}
    for c in candles:
        seen[c.open_time] = c
    if len(seen) == len(candles):
        return candles
    log.warning("removed %d duplicate candle timestamps", len(candles) - len(seen))
    return tuple(sorted(seen.values(), key=lambda c: c.open_time))


def _warn_time_gaps(pair: str, candles: tuple[Candle, ...]) -> None:
    """Log warnings for significant time gaps between candles."""
    if len(candles) < 2:
        return
    expected = (candles[1].open_time - candles[0].open_time).total_seconds()
    if expected <= 0:
        return
    for i in range(1, len(candles)):
        gap = (candles[i].open_time - candles[i - 1].open_time).total_seconds()
        if gap > expected * 2:
            log.warning(
                "%s: time gap of %.0fs (%.1fx expected) between %s and %s",
                pair,
                gap,
                gap / expected,
                candles[i - 1].open_time,
                candles[i].open_time,
            )


def _warn_pair_count_mismatch(pairs: tuple[str, ...], counts: dict[str, int]) -> None:
    """Log warning if pair candle counts differ significantly."""
    if len(pairs) < 2:
        return
    max_count = max(counts.values())
    for pair, count in counts.items():
        if max_count > 0 and count < max_count * 0.8:
            log.warning(
                "%s has %d candles vs max %d (%.0f%% mismatch)",
                pair,
                count,
                max_count,
                (1.0 - count / max_count) * 100,
            )

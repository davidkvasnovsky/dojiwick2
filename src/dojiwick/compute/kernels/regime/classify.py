"""Vectorized deterministic regime classification kernel."""

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import BatchMarketSnapshot, BatchRegimeProfile
from dojiwick.domain.models.value_objects.params import RegimeParams
from dojiwick.domain.enums import MarketState
from dojiwick.domain.indicator_schema import INDICATOR_INDEX

from dojiwick.domain.type_aliases import FloatVector, IntVector

from dojiwick.compute.kernels.math import clamp01


def classify_regime_batch(market: BatchMarketSnapshot, settings: RegimeParams) -> BatchRegimeProfile:
    """Classify coarse market states and confidence for an aligned batch."""

    price = market.price
    indicators = market.indicators

    rsi = indicators[:, INDICATOR_INDEX["rsi"]]
    adx = indicators[:, INDICATOR_INDEX["adx"]]
    atr = indicators[:, INDICATOR_INDEX["atr"]]
    ema_fast = indicators[:, INDICATOR_INDEX["ema_fast"]]
    ema_slow = indicators[:, INDICATOR_INDEX["ema_slow"]]
    ema_base = indicators[:, INDICATOR_INDEX["ema_base"]]
    bb_upper = indicators[:, INDICATOR_INDEX["bb_upper"]]
    bb_lower = indicators[:, INDICATOR_INDEX["bb_lower"]]

    volume_ema_ratio = indicators[:, INDICATOR_INDEX["volume_ema_ratio"]]

    volume_ema_finite = np.isfinite(volume_ema_ratio)

    valid = (
        (price > 0.0)
        & np.isfinite(price)
        & np.isfinite(rsi)
        & np.isfinite(adx)
        & np.isfinite(atr)
        & np.isfinite(ema_fast)
        & np.isfinite(ema_slow)
        & np.isfinite(ema_base)
        & np.isfinite(bb_upper)
        & np.isfinite(bb_lower)
        & (adx >= 0.0)
        & (atr > 0.0)
        & (rsi >= 0.0)
        & (rsi <= 100.0)
        & (bb_upper > bb_lower)
    )

    ema_spread_bps = np.abs(ema_fast - ema_slow) / price * 10_000.0
    atr_pct = atr / price * 100.0

    trend_up = (ema_fast > ema_slow) & (ema_slow > ema_base) & (adx >= settings.adx_trend_min)
    trend_down = (ema_fast < ema_slow) & (ema_slow < ema_base) & (adx >= settings.adx_trend_min)
    trend_up &= ema_spread_bps >= settings.ema_spread_weak_bps
    trend_down &= ema_spread_bps >= settings.ema_spread_weak_bps

    coarse = np.full(price.shape, MarketState.RANGING.value, dtype=np.int64)
    volatile_mask = atr_pct >= settings.atr_high_pct
    coarse[volatile_mask] = MarketState.VOLATILE.value

    up_mask = (~volatile_mask) & trend_up
    down_mask = (~volatile_mask) & (~trend_up) & trend_down
    coarse[up_mask] = MarketState.TRENDING_UP.value
    coarse[down_mask] = MarketState.TRENDING_DOWN.value

    adx_range = max(settings.adx_strong_trend_min - settings.adx_trend_min, 1e-9)
    spread_range = max(settings.ema_spread_strong_bps - settings.ema_spread_weak_bps, 1e-9)
    trend_component = clamp01((adx - settings.adx_trend_min) / adx_range)
    spread_component = clamp01((ema_spread_bps - settings.ema_spread_weak_bps) / spread_range)

    vol_component = np.full(price.shape, 0.5, dtype=np.float64)
    volatile_range = max(settings.atr_extreme_pct - settings.atr_high_pct, 1e-9)
    range_range = max(settings.atr_high_pct - settings.atr_low_pct, 1e-9)

    volatile_rows = coarse == MarketState.VOLATILE.value
    ranging_rows = coarse == MarketState.RANGING.value
    vol_component[volatile_rows] = 0.5 + 0.5 * clamp01(
        (atr_pct[volatile_rows] - settings.atr_high_pct) / volatile_range
    )
    vol_component[ranging_rows] = 1.0 - clamp01((atr_pct[ranging_rows] - settings.atr_low_pct) / range_range)
    trending_rows = ~volatile_rows & ~ranging_rows & valid
    vol_component[trending_rows] = 1.0 - 0.5 * clamp01((atr_pct[trending_rows] - settings.atr_low_pct) / range_range)

    confidence = clamp01(
        settings.trend_weight * trend_component
        + settings.spread_weight * spread_component
        + settings.vol_weight * vol_component
    )

    volume_modifier = np.where(
        volume_ema_finite & (volume_ema_ratio > 0.0),
        np.clip(volume_ema_ratio, settings.volume_clip_lo, settings.volume_clip_hi),
        1.0,
    )
    confidence = clamp01(confidence * volume_modifier)

    confidence[~valid] = 0.0
    coarse[~valid] = MarketState.RANGING.value

    return BatchRegimeProfile(
        coarse_state=coarse,
        confidence=confidence,
        valid_mask=valid.astype(np.bool_),
    )


def truth_labels_from_prices(prices: FloatVector, horizon: int, settings: RegimeParams) -> IntVector:
    """Build deterministic forward truth labels for one horizon."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    labels = np.full(prices.shape, MarketState.RANGING.value, dtype=np.int64)
    if len(prices) <= horizon:
        return labels

    entry = prices[:-horizon]
    future = prices[horizon:]
    change_pct = (future - entry) / entry * 100.0
    abs_change = np.abs(change_pct)

    slice_out = labels[:-horizon]
    slice_out[abs_change >= settings.truth_volatile_return_pct] = MarketState.VOLATILE.value
    up = (abs_change < settings.truth_volatile_return_pct) & (change_pct >= settings.truth_trend_return_pct)
    down = (abs_change < settings.truth_volatile_return_pct) & (change_pct <= -settings.truth_trend_return_pct)
    slice_out[up] = MarketState.TRENDING_UP.value
    slice_out[down] = MarketState.TRENDING_DOWN.value

    return labels

"""Trend-follow strategy signal kernel."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import (
    StrategyParams,
    resolve_optional_param_vector,
    resolve_param_vector,
)
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector
from dojiwick.compute.kernels.strategy._filters import (
    apply_macd_filter,
    apply_volume_filter,
    ema_triple_aligned_down,
    ema_triple_aligned_up,
)


def _resolve_confidence_cap(
    per_pair_settings: tuple[StrategyParams, ...] | None,
    pre_extracted: dict[str, np.ndarray] | None,
    settings_value: float | None,
    param_name: str,
) -> np.ndarray | float | None:
    """Resolve a confidence cap from per-pair overrides or global settings."""
    if per_pair_settings is not None:
        return resolve_optional_param_vector(pre_extracted, per_pair_settings, param_name, default=0.0)
    return settings_value


def _apply_confidence_cap(
    mask: BoolVector,
    regime_confidence: FloatVector | None,
    max_conf: np.ndarray | float | None,
) -> BoolVector:
    """Suppress signals where regime confidence exceeds the cap."""
    if regime_confidence is None or max_conf is None:
        return mask
    if isinstance(max_conf, np.ndarray):
        too_confident = (max_conf > 0.0) & (regime_confidence > max_conf)
    else:
        too_confident = regime_confidence > max_conf
    return mask & ~too_confident


def trend_follow_signal(
    *,
    states: IntVector,
    indicators: FloatMatrix,
    prices: FloatVector,
    settings: StrategyParams,
    per_pair_settings: tuple[StrategyParams, ...] | None = None,
    pre_extracted: dict[str, np.ndarray] | None = None,
    regime_confidence: FloatVector | None = None,
) -> tuple[BoolVector, BoolVector]:
    """Return masks for trend-follow buy and short signals."""

    rsi = indicators[:, INDICATOR_INDEX["rsi"]]
    adx = indicators[:, INDICATOR_INDEX["adx"]]
    ema_fast = indicators[:, INDICATOR_INDEX["ema_fast"]]
    ema_slow = indicators[:, INDICATOR_INDEX["ema_slow"]]
    ema_base = indicators[:, INDICATOR_INDEX["ema_base"]]
    bb_upper = indicators[:, INDICATOR_INDEX["bb_upper"]]
    bb_lower = indicators[:, INDICATOR_INDEX["bb_lower"]]
    ema_trend = indicators[:, INDICATOR_INDEX["ema_trend"]]
    volume_ema_ratio = indicators[:, INDICATOR_INDEX["volume_ema_ratio"]]

    trending_up = states == MarketState.TRENDING_UP.value
    trending_down = states == MarketState.TRENDING_DOWN.value

    # Volatile-trend detection: high ATR periods with clear EMA alignment
    # are eligible for trend entries (BTC is often volatile AND trending).
    # NOTE: this is a global toggle read from ``settings`` (not per-pair).
    # Scope overrides for ``trend_volatile_ema_enabled`` have no effect here.
    if settings.trend_volatile_ema_enabled:
        volatile = states == MarketState.VOLATILE.value
        ema_aligned_up = volatile & ema_triple_aligned_up(ema_fast, ema_slow, ema_base)
        ema_aligned_down = volatile & ema_triple_aligned_down(ema_fast, ema_slow, ema_base)
        trending_up = trending_up | ema_aligned_up
        trending_down = trending_down | ema_aligned_down

    if per_pair_settings is not None:
        pullback_rsi_max = resolve_param_vector(pre_extracted, per_pair_settings, "trend_pullback_rsi_max")
        overbought_rsi_min = resolve_param_vector(pre_extracted, per_pair_settings, "trend_overbought_rsi_min")
        breakout_adx_min = resolve_param_vector(pre_extracted, per_pair_settings, "trend_breakout_adx_min")
    else:
        pullback_rsi_max = settings.trend_pullback_rsi_max
        overbought_rsi_min = settings.trend_overbought_rsi_min
        breakout_adx_min = settings.trend_breakout_adx_min

    buy_pullback = (
        trending_up & (rsi <= pullback_rsi_max) & (prices > ema_base) & (ema_fast > ema_slow) & (prices > ema_trend)
    )

    # ADX pullback filter — reject weak pullbacks when ADX is too low
    pullback_adx_ok: np.ndarray | None = None
    if per_pair_settings is not None:
        pullback_adx_min = resolve_optional_param_vector(
            pre_extracted,
            per_pair_settings,
            "trend_pullback_adx_min",
            default=0.0,
        )
        if pullback_adx_min is not None:
            pullback_adx_ok = (pullback_adx_min == 0.0) | (adx >= pullback_adx_min)
    elif settings.trend_pullback_adx_min is not None:
        pullback_adx_ok = adx >= settings.trend_pullback_adx_min

    if pullback_adx_ok is not None:
        buy_pullback = buy_pullback & pullback_adx_ok

    buy_breakout = trending_up & (prices >= bb_upper) & (adx >= breakout_adx_min) & (prices > ema_trend)
    buy_mask = buy_pullback | buy_breakout

    short_pullback = (
        trending_down & (rsi >= overbought_rsi_min) & (prices < ema_base) & (ema_fast < ema_slow) & (prices < ema_trend)
    )

    if pullback_adx_ok is not None:
        short_pullback = short_pullback & pullback_adx_ok

    short_breakdown = trending_down & (prices <= bb_lower) & (adx >= breakout_adx_min) & (prices < ema_trend)
    short_mask = short_pullback | short_breakdown

    buy_conf = _resolve_confidence_cap(
        per_pair_settings, pre_extracted, settings.trend_max_regime_confidence, "trend_max_regime_confidence"
    )
    buy_mask = _apply_confidence_cap(buy_mask, regime_confidence, buy_conf)

    short_conf = _resolve_confidence_cap(
        per_pair_settings,
        pre_extracted,
        settings.trend_short_max_regime_confidence,
        "trend_short_max_regime_confidence",
    )
    short_mask = _apply_confidence_cap(short_mask, regime_confidence, short_conf)

    if settings.macd_filter_enabled:
        buy_mask, short_mask = apply_macd_filter(buy_mask, short_mask, indicators)

    return apply_volume_filter(buy_mask, short_mask, volume_ema_ratio, settings, per_pair_settings, pre_extracted)

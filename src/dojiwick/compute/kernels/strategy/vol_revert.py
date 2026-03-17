"""Volatility-reversion strategy signal kernel."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import StrategyParams, resolve_param_vector
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector
from dojiwick.compute.kernels.strategy._filters import apply_macd_filter, apply_volume_filter


def vol_revert_signal(
    *,
    states: IntVector,
    indicators: FloatMatrix,
    prices: FloatVector,
    settings: StrategyParams,
    per_pair_settings: tuple[StrategyParams, ...] | None = None,
    pre_extracted: dict[str, np.ndarray] | None = None,
    regime_confidence: FloatVector | None = None,
) -> tuple[BoolVector, BoolVector]:
    """Return masks for volatility-reversion buy and short signals."""

    rsi = indicators[:, INDICATOR_INDEX["rsi"]]
    bb_lower = indicators[:, INDICATOR_INDEX["bb_lower"]]
    bb_upper = indicators[:, INDICATOR_INDEX["bb_upper"]]
    ema_slow = indicators[:, INDICATOR_INDEX["ema_slow"]]
    ema_trend = indicators[:, INDICATOR_INDEX["ema_trend"]]
    volume_ema_ratio = indicators[:, INDICATOR_INDEX["volume_ema_ratio"]]
    volatile = states == MarketState.VOLATILE.value

    if per_pair_settings is not None:
        oversold = resolve_param_vector(pre_extracted, per_pair_settings, "vol_extreme_oversold")
        overbought = resolve_param_vector(pre_extracted, per_pair_settings, "vol_extreme_overbought")
    else:
        oversold = settings.vol_extreme_oversold
        overbought = settings.vol_extreme_overbought

    buy_mask = volatile & (rsi <= oversold) & (prices <= bb_lower) & (ema_slow > ema_trend)
    short_mask = volatile & (rsi >= overbought) & (prices >= bb_upper) & (ema_slow < ema_trend)

    if settings.macd_filter_enabled:
        buy_mask, short_mask = apply_macd_filter(buy_mask, short_mask, indicators)

    return apply_volume_filter(buy_mask, short_mask, volume_ema_ratio, settings, per_pair_settings, pre_extracted)

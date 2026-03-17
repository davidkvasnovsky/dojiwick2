"""Mean-reversion strategy signal kernel."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import StrategyParams, resolve_param_vector
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector
from dojiwick.compute.kernels.strategy._filters import apply_volume_filter


def mean_revert_signal(
    *,
    states: IntVector,
    indicators: FloatMatrix,
    prices: FloatVector,
    settings: StrategyParams,
    per_pair_settings: tuple[StrategyParams, ...] | None = None,
    pre_extracted: dict[str, np.ndarray] | None = None,
    regime_confidence: FloatVector | None = None,
) -> tuple[BoolVector, BoolVector]:
    """Return masks for mean-revert buy and short signals."""

    rsi = indicators[:, INDICATOR_INDEX["rsi"]]
    bb_upper = indicators[:, INDICATOR_INDEX["bb_upper"]]
    bb_lower = indicators[:, INDICATOR_INDEX["bb_lower"]]
    ema_slow = indicators[:, INDICATOR_INDEX["ema_slow"]]
    ema_trend = indicators[:, INDICATOR_INDEX["ema_trend"]]
    volume_ema_ratio = indicators[:, INDICATOR_INDEX["volume_ema_ratio"]]

    ranging = states == MarketState.RANGING.value

    if per_pair_settings is not None:
        oversold = resolve_param_vector(pre_extracted, per_pair_settings, "mean_rsi_oversold")
        overbought = resolve_param_vector(pre_extracted, per_pair_settings, "mean_rsi_overbought")
    else:
        oversold = settings.mean_rsi_oversold
        overbought = settings.mean_rsi_overbought

    if settings.mean_revert_disable_ema_filter:
        buy_mask = ranging & (rsi <= oversold) & (prices <= bb_lower)
        short_mask = ranging & (rsi >= overbought) & (prices >= bb_upper)
    else:
        buy_mask = ranging & (rsi <= oversold) & (prices <= bb_lower) & (ema_slow > ema_trend)
        short_mask = ranging & (rsi >= overbought) & (prices >= bb_upper) & (ema_slow < ema_trend)

    return apply_volume_filter(buy_mask, short_mask, volume_ema_ratio, settings, per_pair_settings, pre_extracted)

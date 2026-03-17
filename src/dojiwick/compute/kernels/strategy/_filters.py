"""Shared signal filters for strategy kernels."""

import numpy as np

from dojiwick.domain.models.value_objects.params import StrategyParams, resolve_param_vector
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector


def ema_triple_aligned_up(ema_fast: FloatVector, ema_slow: FloatVector, ema_base: FloatVector) -> BoolVector:
    """True where fast > slow > base (bullish alignment)."""
    return (ema_fast > ema_slow) & (ema_slow > ema_base)


def ema_triple_aligned_down(ema_fast: FloatVector, ema_slow: FloatVector, ema_base: FloatVector) -> BoolVector:
    """True where fast < slow < base (bearish alignment)."""
    return (ema_fast < ema_slow) & (ema_slow < ema_base)


def macd_direction_aligned(macd_hist: FloatVector, buy: BoolVector, short: BoolVector) -> BoolVector:
    """True where MACD histogram confirms the trade direction. HOLD rows return False."""
    return (buy & (macd_hist > 0)) | (short & (macd_hist < 0))


def apply_macd_filter(
    buy_mask: BoolVector,
    short_mask: BoolVector,
    indicators: FloatMatrix,
) -> tuple[BoolVector, BoolVector]:
    """Suppress entries against MACD histogram direction."""
    macd_hist = indicators[:, INDICATOR_INDEX["macd_histogram"]]
    return buy_mask & (macd_hist > 0.0), short_mask & (macd_hist < 0.0)


def apply_volume_filter(
    buy_mask: BoolVector,
    short_mask: BoolVector,
    volume_ema_ratio: FloatVector,
    settings: StrategyParams,
    per_pair_settings: tuple[StrategyParams, ...] | None = None,
    pre_extracted: dict[str, np.ndarray] | None = None,
) -> tuple[BoolVector, BoolVector]:
    """Reject entries on below-average volume bars."""
    if per_pair_settings is not None:
        min_vol = resolve_param_vector(pre_extracted, per_pair_settings, "min_volume_ratio")
    else:
        min_vol = settings.min_volume_ratio
    vol_ok = volume_ema_ratio >= min_vol
    return buy_mask & vol_ok, short_mask & vol_ok

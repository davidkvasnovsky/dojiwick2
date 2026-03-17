"""Entry confluence scoring kernel — quality gate for trade entries.

Scores each entry on 5 dimensions (0-100 total). Only entries above the
configured threshold are allowed through.
"""

import numpy as np

from dojiwick.compute.kernels.strategy._filters import (
    ema_triple_aligned_down,
    ema_triple_aligned_up,
    macd_direction_aligned,
)
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.models.value_objects.params import StrategyParams
from dojiwick.domain.type_aliases import FloatMatrix, FloatVector, IntVector


def compute_confluence_score(
    indicators: FloatMatrix,
    prices: FloatVector,
    action: IntVector,
    settings: StrategyParams,
) -> FloatVector:
    """Score entry quality on 5 dimensions (0-100 total)."""
    score = np.zeros(len(prices), dtype=np.float64)

    rsi = indicators[:, INDICATOR_INDEX["rsi"]]
    adx = indicators[:, INDICATOR_INDEX["adx"]]
    macd_hist = indicators[:, INDICATOR_INDEX["macd_histogram"]]
    volume_ratio = indicators[:, INDICATOR_INDEX["volume_ema_ratio"]]
    ema_fast = indicators[:, INDICATOR_INDEX["ema_fast"]]
    ema_slow = indicators[:, INDICATOR_INDEX["ema_slow"]]
    ema_base = indicators[:, INDICATOR_INDEX["ema_base"]]

    buy = action == TradeAction.BUY.value
    short = action == TradeAction.SHORT.value
    active = buy | short

    # 1. RSI confirmation (0-20 pts)
    rsi_mid = settings.confluence_rsi_midpoint
    rsi_rng = settings.confluence_rsi_range
    rsi_score = np.where(
        buy,
        np.clip((rsi_mid - rsi) / rsi_rng * 20, 0, 20),
        np.where(short, np.clip((rsi - rsi_mid) / rsi_rng * 20, 0, 20), 0),
    )
    score += rsi_score

    # 2. MACD momentum alignment (0-20 pts)
    macd_ok = macd_direction_aligned(macd_hist, buy, short)
    score += np.where(macd_ok, 20.0, 0.0)

    # 3. Volume surge (0-20 pts)
    vol_base = settings.confluence_volume_baseline
    vol_mult = settings.confluence_volume_multiplier
    score += np.where(active, np.clip((volume_ratio - vol_base) * vol_mult, 0, 20), 0.0)

    # 4. EMA alignment (0-20 pts)
    ema_aligned = np.where(
        buy,
        ema_triple_aligned_up(ema_fast, ema_slow, ema_base),
        np.where(short, ema_triple_aligned_down(ema_fast, ema_slow, ema_base), False),
    )
    score += np.where(ema_aligned, 20.0, 0.0)

    # 5. ADX trend strength (0-20 pts)
    adx_base = settings.confluence_adx_baseline
    adx_rng = settings.confluence_adx_range
    score += np.where(active, np.clip((adx - adx_base) / adx_rng * 20, 0, 20), 0.0)

    return score

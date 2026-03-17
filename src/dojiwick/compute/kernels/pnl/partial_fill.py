"""Partial fill simulation kernel for backtest fill modeling.

Computes fill ratios based on order size relative to bar volume,
then scales quantity and notional to simulate partial execution.
"""

import numpy as np

from dojiwick.domain.enums import TradeAction
from dojiwick.domain.type_aliases import FloatVector, IntVector


def compute_fill_ratio(
    *,
    notional_usd: FloatVector,
    bar_volume: FloatVector,
    entry_price: FloatVector,
    action: IntVector,
    threshold_pct: float,
    min_ratio: float,
) -> FloatVector:
    """Compute fill ratio per pair based on order size vs bar volume.

    Orders whose notional exceeds ``threshold_pct`` of bar volume
    (in USD terms) receive partial fills.  HOLD actions always get 1.0.

    Parameters
    ----------
    notional_usd:
        Order notional per pair (N).
    bar_volume:
        Bar volume in base units per pair (N).
    entry_price:
        Entry price per pair (N).
    action:
        Trade action per pair (N).
    threshold_pct:
        Fraction of bar volume USD above which orders are partially filled.
    min_ratio:
        Minimum fill ratio (floor).

    Returns
    -------
    FloatVector
        Fill ratio in [min_ratio, 1.0] per pair.
    """
    bar_volume_usd = bar_volume * entry_price
    # Avoid division by zero
    safe_notional = np.where(notional_usd > 0, notional_usd, 1.0)
    raw_ratio = (threshold_pct * bar_volume_usd) / safe_notional
    fill_ratio = np.clip(raw_ratio, min_ratio, 1.0)

    # HOLD actions always get 1.0
    hold_mask = action == TradeAction.HOLD.value
    fill_ratio[hold_mask] = 1.0

    return fill_ratio


def apply_fill_ratio(
    *,
    quantity: FloatVector,
    notional_usd: FloatVector,
    fill_ratio: FloatVector,
) -> tuple[FloatVector, FloatVector]:
    """Scale quantity and notional by fill ratio.

    Returns
    -------
    tuple[FloatVector, FloatVector]
        (filled_quantity, filled_notional)
    """
    return quantity * fill_ratio, notional_usd * fill_ratio

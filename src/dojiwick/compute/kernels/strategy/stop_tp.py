"""ATR-based stop/take-profit placement — single implementation for batch and scalar callers."""

import numpy as np

from dojiwick.domain.type_aliases import FloatVector, IntVector


def atr_stop_take_profit(
    entry: FloatVector,
    atr: FloatVector,
    direction: IntVector,
    stop_atr_mult: FloatVector,
    rr_ratio: FloatVector,
    min_stop_pct: FloatVector,
) -> tuple[FloatVector, FloatVector]:
    """Stop and take-profit from ATR distance with a percent floor.

    ``direction`` is +1 for long, -1 for short, 0 for no position (row stays 0).
    Distance = max(atr * stop_atr_mult, entry * min_stop_pct / 100).
    """
    distance = np.maximum(atr * stop_atr_mult, entry * min_stop_pct / 100.0)
    active = direction != 0
    stop = np.where(active, entry - direction * distance, 0.0)
    take_profit = np.where(active, entry + direction * distance * rr_ratio, 0.0)
    return np.maximum(stop, 0.0), np.maximum(take_profit, 0.0)


def atr_stop_take_profit_scalar(
    entry: float,
    atr: float,
    direction: int,
    stop_atr_mult: float,
    rr_ratio: float,
    min_stop_pct: float,
) -> tuple[float, float]:
    """Scalar wrapper over :func:`atr_stop_take_profit` for single-pair callers."""
    stop, take_profit = atr_stop_take_profit(
        np.asarray(entry, dtype=np.float64),
        np.asarray(atr, dtype=np.float64),
        np.asarray(direction, dtype=np.int64),
        np.asarray(stop_atr_mult, dtype=np.float64),
        np.asarray(rr_ratio, dtype=np.float64),
        np.asarray(min_stop_pct, dtype=np.float64),
    )
    return float(stop), float(take_profit)

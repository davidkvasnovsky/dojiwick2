"""Scalar entry-risk scaling — the single source for ECF and drawdown scaling.

Both the backtest bar loop and the live tick derive new-entry size multipliers
from these formulas; keeping them here stops the two paths from drifting.
"""


def equity_curve_scale(equity: float, sma: float, floor: float) -> float:
    """Equity-curve filter: proportional size reduction when equity < its SMA.

    Returns 1.0 (no reduction) when equity is at or above the SMA. Never blocks;
    the reduction is floored at ``floor``.
    """
    if sma > 0 and equity < sma:
        return max(equity / sma, floor)
    return 1.0


def drawdown_scale(drawdown_pct: float, max_dd_pct: float, floor: float) -> float:
    """Sqrt-curve size reduction as portfolio drawdown deepens, floored at ``floor``.

    Returns 1.0 when not in drawdown. ``drawdown_pct`` and ``max_dd_pct`` are on
    the 0-100 scale.
    """
    if drawdown_pct <= 0:
        return 1.0
    raw = max(1.0 - drawdown_pct / max_dd_pct, 0.0)
    return max(raw**0.5, floor)

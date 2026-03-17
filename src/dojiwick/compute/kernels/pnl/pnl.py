"""PnL and fee kernels shared by backtest and optimization."""

import numpy as np

from dojiwick.domain.enums import TradeAction
from dojiwick.domain.type_aliases import FloatVector, IntVector


def apply_slippage(
    entry_price: FloatVector,
    action: IntVector,
    slippage_bps: float,
    *,
    notional: FloatVector | None = None,
    volume: FloatVector | None = None,
    impact_bps: float = 0.0,
) -> FloatVector:
    """Apply directional slippage to entry prices.

    When volume and notional are provided, adds volume-scaled market impact:
    ``slippage = base_bps + (notional / volume) * impact_bps``
    """
    if impact_bps > 0.0 and notional is not None and volume is not None:
        safe_volume = np.where(volume > 0.0, volume, 1.0)
        volume_impact = (notional / safe_volume) * impact_bps
        effective_bps = slippage_bps + volume_impact
        slip = entry_price * (effective_bps / 10_000.0)
    else:
        slip = entry_price * (slippage_bps / 10_000.0)
    adjusted = entry_price.copy()

    buy_rows = action == TradeAction.BUY.value
    short_rows = action == TradeAction.SHORT.value

    adjusted[buy_rows] = entry_price[buy_rows] + slip[buy_rows]
    adjusted[short_rows] = entry_price[short_rows] - slip[short_rows]
    return adjusted


def gross_pnl(
    action: IntVector,
    entry: FloatVector,
    exit_price: FloatVector,
    quantity: FloatVector,
    leverage: float = 1.0,
) -> FloatVector:
    """Compute gross pnl for long and short rows, scaled by leverage."""

    pnl = np.zeros_like(entry, dtype=np.float64)
    buy_rows = action == TradeAction.BUY.value
    short_rows = action == TradeAction.SHORT.value

    pnl[buy_rows] = (exit_price[buy_rows] - entry[buy_rows]) * quantity[buy_rows]
    pnl[short_rows] = (entry[short_rows] - exit_price[short_rows]) * quantity[short_rows]
    return pnl * leverage


def net_pnl(
    gross: FloatVector,
    notional: FloatVector,
    fee_bps: float,
    fee_multiplier: float = 2.0,
    *,
    funding_rate_per_bar: float = 0.0,
    action: IntVector | None = None,
    hold_bars: int | IntVector = 1,
) -> FloatVector:
    """Subtract open+close fees and funding costs from gross PnL."""

    fee_rate = fee_bps / 10_000.0
    fees = notional * fee_rate * fee_multiplier
    result = gross - fees
    if funding_rate_per_bar > 0.0 and action is not None:
        active = action != TradeAction.HOLD.value
        result[active] -= notional[active] * funding_rate_per_bar * hold_bars
    return result


def scalar_net_pnl(
    is_long: bool,
    entry_price: float,
    exit_price: float,
    quantity: float,
    notional: float,
    slippage_bps: float,
    fee_bps: float,
    fee_multiplier: float = 2.0,
    leverage: float = 1.0,
    funding_rate_per_bar: float = 0.0,
    hold_bars: int = 1,
) -> float:
    """Scalar net PnL for a single closed position — single source of truth."""
    slip = entry_price * (slippage_bps / 10_000.0)
    slipped = entry_price + slip if is_long else entry_price - slip
    gross = ((exit_price - slipped) if is_long else (slipped - exit_price)) * quantity * leverage
    fees = notional * (fee_bps / 10_000.0) * fee_multiplier
    funding = notional * funding_rate_per_bar * max(hold_bars, 1) if funding_rate_per_bar > 0.0 else 0.0
    return gross - fees - funding

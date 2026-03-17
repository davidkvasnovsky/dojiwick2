"""Liquidation modeling for leveraged backtest positions."""

import numpy as np

from dojiwick.domain.enums import TradeAction
from dojiwick.domain.type_aliases import BoolVector, FloatVector, IntVector


def check_liquidation(
    entry: FloatVector,
    next_price: FloatVector,
    leverage: float,
    action: IntVector,
    maintenance_margin_rate: float = 0.0,
) -> BoolVector:
    """Check which positions would be liquidated at given leverage.

    When ``maintenance_margin_rate == 0`` (legacy mode):
        Longs: liquidated when next_price <= entry * (1 - 1/leverage)
        Shorts: liquidated when next_price >= entry * (1 + 1/leverage)

    When ``maintenance_margin_rate > 0`` (exchange-realistic):
        initial_margin_rate = 1 / leverage
        Longs: liquidated when next_price <= entry * (1 - (IMR - MMR))
        Shorts: liquidated when next_price >= entry * (1 + (IMR - MMR))

    Only active when leverage > 1.0.
    """
    liquidated = np.zeros(len(entry), dtype=np.bool_)
    if leverage <= 1.0:
        return liquidated

    initial_margin_rate = 1.0 / leverage
    if maintenance_margin_rate > 0.0:
        margin_distance = initial_margin_rate - maintenance_margin_rate
    else:
        margin_distance = initial_margin_rate

    buy_rows = action == TradeAction.BUY.value
    short_rows = action == TradeAction.SHORT.value

    liq_price_long = entry * (1.0 - margin_distance)
    liq_price_short = entry * (1.0 + margin_distance)

    liquidated[buy_rows] = next_price[buy_rows] <= liq_price_long[buy_rows]
    liquidated[short_rows] = next_price[short_rows] >= liq_price_short[short_rows]
    return liquidated


def cap_pnl_at_margin(
    pnl: FloatVector,
    notional: FloatVector,
    leverage: float,
    liquidated: BoolVector,
) -> FloatVector:
    """Cap loss at margin for liquidated positions."""
    if leverage <= 1.0 or not np.any(liquidated):
        return pnl
    result = pnl.copy()
    margin = notional[liquidated] / leverage
    result[liquidated] = np.maximum(result[liquidated], -margin)
    return result

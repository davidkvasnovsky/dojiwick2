"""Entry price resolution kernel for backtest fill simulation.

Resolves entry price based on configured model: close (default),
next bar open, VWAP proxy, or worst-case (adversarial).
"""

from dojiwick.domain.enums import EntryPriceModel, TradeAction
from dojiwick.domain.type_aliases import FloatVector, IntVector


def resolve_entry_price(
    model: EntryPriceModel,
    *,
    close: FloatVector,
    next_open: FloatVector,
    next_high: FloatVector,
    next_low: FloatVector,
    next_close: FloatVector,
    action: IntVector,
) -> FloatVector:
    """Resolve entry price for each pair based on the configured model.

    Parameters
    ----------
    model:
        Entry price model to use.
    close:
        Current bar close prices (N pairs).
    next_open, next_high, next_low, next_close:
        Next bar OHLC prices (N pairs).
    action:
        Trade action per pair (BUY=1, SHORT=-1, HOLD=0).

    Returns
    -------
    FloatVector
        Resolved entry prices (N pairs).
    """
    if model == EntryPriceModel.CLOSE:
        return close.copy()

    if model == EntryPriceModel.NEXT_OPEN:
        return next_open.copy()

    if model == EntryPriceModel.VWAP_PROXY:
        return (next_open + next_high + next_low + next_close) / 4.0

    # worst_case: next_high for longs, next_low for shorts, close for holds
    result = close.copy()
    buy_mask = action == TradeAction.BUY.value
    short_mask = action == TradeAction.SHORT.value
    result[buy_mask] = next_high[buy_mask]
    result[short_mask] = next_low[short_mask]
    return result

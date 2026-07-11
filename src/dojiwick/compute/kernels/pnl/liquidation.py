"""Liquidation price for leveraged backtest positions."""


def liquidation_price(
    entry_price: float,
    leverage: float,
    maintenance_margin_rate: float,
    is_long: bool,
) -> float:
    """Price at which the position's margin is exhausted, or 0.0 when unleveraged.

    ``margin_distance = 1/leverage - maintenance_margin_rate``: the adverse move
    that consumes initial margin down to the maintenance requirement, at which
    point the exchange force-closes the position.
    """
    if leverage <= 1.0:
        return 0.0
    margin_distance = 1.0 / leverage - maintenance_margin_rate
    if is_long:
        return entry_price * (1.0 - margin_distance)
    return entry_price * (1.0 + margin_distance)

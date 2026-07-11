"""Scalar PnL kernel shared by backtest and optimization."""


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
    funding_usd: float = 0.0,
) -> float:
    """Net PnL for a closed position tranche.

    ``notional`` is margin-based (quantity x entry price); the exchange
    position is ``notional x leverage``, so fees are charged on the
    leveraged value just as gross PnL scales with the leveraged quantity.
    Slippage is adverse on both entry and exit fills. ``funding_usd`` is
    the signed accrued funding cost (positive = paid).
    """
    slip_rate = slippage_bps / 10_000.0
    if is_long:
        entry_eff = entry_price * (1.0 + slip_rate)
        exit_eff = exit_price * (1.0 - slip_rate)
        gross = (exit_eff - entry_eff) * quantity * leverage
    else:
        entry_eff = entry_price * (1.0 - slip_rate)
        exit_eff = exit_price * (1.0 + slip_rate)
        gross = (entry_eff - exit_eff) * quantity * leverage
    fees = notional * leverage * (fee_bps / 10_000.0) * fee_multiplier
    return gross - fees - funding_usd

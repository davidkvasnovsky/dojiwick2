"""Cost model for PnL computation."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class CostModel:
    """Bundled cost parameters for backtest PnL kernels."""

    fee_bps: float = 4.0
    fee_multiplier: float = 2.0
    slippage_bps: float = 2.0
    leverage: float = 1.0
    maintenance_margin_rate: float = 0.0

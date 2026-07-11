"""Mutable exit-management state for a live position leg.

The live twin of the backtest's ``_OpenPosition`` exit fields: everything the
protective-order reconciler needs to derive, amend, and retire STOP/TP orders
for one leg. ``revision`` feeds deterministic protective client-order ids so
an amended stop gets a fresh id while an unchanged one stays idempotent.
"""

from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class PositionExitState:
    position_leg_id: int
    is_long: bool
    entry_price: float
    stop_price: float
    original_stop: float
    take_profit_price: float
    trailing_activation_price: float = 0.0
    trailing_distance: float = 0.0
    breakeven_price: float = 0.0
    extreme_price: float = 0.0
    max_hold_bars: int = 0
    bars_held: int = 0
    tp1_price: float = 0.0
    tp1_fraction: float = 0.0
    tp1_filled: bool = False
    revision: int = 0

    def __post_init__(self) -> None:
        if self.position_leg_id <= 0:
            raise ValueError("position_leg_id must be positive")
        if self.entry_price <= 0 or self.stop_price <= 0 or self.take_profit_price <= 0:
            raise ValueError("entry/stop/take-profit prices must be positive")

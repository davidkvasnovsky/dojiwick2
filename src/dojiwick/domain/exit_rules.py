"""Pure exit-management rules shared by the backtest and live protective orders.

``advance_trailing_stop`` and ``derive_exit_anchors`` are the single source of
the trailing/breakeven/TP1 math; ``run_backtest`` and the live tick both call
them so the two paths cannot drift.
"""

from dataclasses import dataclass

from dojiwick.domain.models.entities.position_exit_state import PositionExitState
from dojiwick.domain.models.value_objects.params import StrategyParams


def advance_trailing_stop(
    *,
    is_long: bool,
    high: float,
    low: float,
    extreme_price: float,
    stop_price: float,
    entry_price: float,
    trailing_activation_price: float,
    trailing_distance: float,
    breakeven_price: float,
) -> tuple[float, float]:
    """Return ``(new_extreme, new_stop)`` after breakeven/trailing advancement."""
    new_extreme = max(extreme_price, high) if is_long else min(extreme_price, low)
    new_stop = stop_price

    # Breakeven: once price exceeds breakeven threshold, move stop to entry
    be_hit = (new_extreme >= breakeven_price) if is_long else (new_extreme <= breakeven_price)
    stop_not_at_entry = (stop_price < entry_price) if is_long else (stop_price > entry_price)
    if breakeven_price > 0.0 and be_hit and stop_not_at_entry:
        new_stop = entry_price

    # Trailing: once price exceeds activation, trail stop behind extreme
    act_hit = (new_extreme >= trailing_activation_price) if is_long else (new_extreme <= trailing_activation_price)
    if trailing_activation_price > 0.0 and act_hit:
        trail_stop = (new_extreme - trailing_distance) if is_long else (new_extreme + trailing_distance)
        new_stop = max(new_stop, trail_stop) if is_long else min(new_stop, trail_stop)

    return new_extreme, new_stop


def update_trailing_stop(state: PositionExitState, high: float, low: float, is_long: bool) -> None:
    """Advance extreme price and trailing/breakeven stop in place."""
    if state.trailing_activation_price == 0.0 and state.breakeven_price == 0.0:
        return
    new_extreme, new_stop = advance_trailing_stop(
        is_long=is_long,
        high=high,
        low=low,
        extreme_price=state.extreme_price,
        stop_price=state.stop_price,
        entry_price=state.entry_price,
        trailing_activation_price=state.trailing_activation_price,
        trailing_distance=state.trailing_distance,
        breakeven_price=state.breakeven_price,
    )
    state.extreme_price = new_extreme
    if new_stop != state.stop_price:
        state.stop_price = new_stop
        state.revision += 1


def should_time_exit(state: PositionExitState) -> bool:
    """True when the position exceeded its maximum hold duration."""
    return state.max_hold_bars > 0 and state.bars_held >= state.max_hold_bars


@dataclass(slots=True, frozen=True, kw_only=True)
class ExitAnchors:
    """Trailing/breakeven/TP1 anchors derived from an entry fill."""

    trailing_activation_price: float = 0.0
    trailing_distance: float = 0.0
    breakeven_price: float = 0.0
    max_hold_bars: int = 0
    tp1_price: float = 0.0
    tp1_fraction: float = 0.0


def derive_exit_anchors(
    *,
    entry_price: float,
    stop_distance: float,
    direction: float,
    params: StrategyParams,
    atr: float | None,
) -> ExitAnchors:
    """Derive exit anchors from entry geometry — shared by backtest and live.

    ``direction`` is +1 for long, -1 for short. ``atr`` is the ATR at entry;
    ``None`` (live path — ATR is not carried on the intent) derives the trail
    distance from the stop geometry, which is itself ATR-based.
    """
    trailing_activation = 0.0
    trailing_distance = 0.0
    if params.trailing_stop_activation_rr is not None and params.trailing_stop_atr_mult is not None:
        trailing_activation = entry_price + direction * stop_distance * params.trailing_stop_activation_rr
        if atr is not None:
            trailing_distance = atr * params.trailing_stop_atr_mult
        else:
            trailing_distance = stop_distance * params.trailing_stop_atr_mult / max(params.stop_atr_mult, 1e-9)

    breakeven = 0.0
    if params.breakeven_after_rr is not None:
        breakeven = entry_price + direction * stop_distance * params.breakeven_after_rr

    tp1_price = 0.0
    tp1_fraction = 0.0
    if params.partial_tp_enabled and params.partial_tp1_rr > 0:
        tp1_price = entry_price + direction * stop_distance * params.partial_tp1_rr
        tp1_fraction = params.partial_tp1_fraction

    return ExitAnchors(
        trailing_activation_price=trailing_activation,
        trailing_distance=trailing_distance,
        breakeven_price=breakeven,
        max_hold_bars=params.max_hold_bars if params.max_hold_bars is not None else 0,
        tp1_price=tp1_price,
        tp1_fraction=tp1_fraction,
    )

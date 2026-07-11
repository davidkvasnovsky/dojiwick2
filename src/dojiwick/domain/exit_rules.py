"""Pure exit-management rules shared by live protective orders and tests.

Port of the backtest's trailing/breakeven/time-exit logic onto the live
``PositionExitState`` entity — parity is enforced by property tests against
``run_backtest._update_trailing_stop``.
"""

from dojiwick.domain.models.entities.position_exit_state import PositionExitState


def update_trailing_stop(state: PositionExitState, high: float, low: float, is_long: bool) -> None:
    """Advance extreme price and trailing/breakeven stop in place."""
    if state.trailing_activation_price == 0.0 and state.breakeven_price == 0.0:
        return
    new_extreme = max(state.extreme_price, high) if is_long else min(state.extreme_price, low)
    new_stop = state.stop_price

    be_hit = (new_extreme >= state.breakeven_price) if is_long else (new_extreme <= state.breakeven_price)
    stop_not_at_entry = (state.stop_price < state.entry_price) if is_long else (state.stop_price > state.entry_price)
    if state.breakeven_price > 0.0 and be_hit and stop_not_at_entry:
        new_stop = state.entry_price

    act_hit = (
        (new_extreme >= state.trailing_activation_price)
        if is_long
        else (new_extreme <= state.trailing_activation_price)
    )
    if state.trailing_activation_price > 0.0 and act_hit:
        trail_stop = (new_extreme - state.trailing_distance) if is_long else (new_extreme + state.trailing_distance)
        new_stop = max(new_stop, trail_stop) if is_long else min(new_stop, trail_stop)

    state.extreme_price = new_extreme
    if new_stop != state.stop_price:
        state.stop_price = new_stop
        state.revision += 1


def should_time_exit(state: PositionExitState) -> bool:
    """True when the position exceeded its maximum hold duration."""
    return state.max_hold_bars > 0 and state.bars_held >= state.max_hold_bars

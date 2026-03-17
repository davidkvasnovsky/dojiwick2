"""Exchange filter enforcement kernel — truncates quantities/prices to exchange rules."""

import math

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter


def _truncate_to_step(value: float, step: float) -> float:
    """Truncate value down to nearest step_size multiple."""
    if step <= 0:
        return value
    return math.floor(value / step) * step


def _round_to_tick(value: float, tick: float) -> float:
    """Round price to nearest tick_size."""
    if tick <= 0:
        return value
    return round(value / tick) * tick


def apply_exchange_filters(
    intents: BatchExecutionIntent,
    filters: dict[str, InstrumentFilter] | None,
) -> BatchExecutionIntent:
    """Truncate quantities to step_size, enforce min/max qty, exchange min notional.

    No-op when filters is None (scaffold/backtest mode).
    """
    if filters is None:
        return intents

    quantity = intents.quantity.copy()
    entry_price = intents.entry_price.copy()
    stop_price = intents.stop_price.copy()
    take_profit_price = intents.take_profit_price.copy()
    active = intents.active_mask.copy()
    notional = intents.notional_usd.copy()

    for i, pair in enumerate(intents.pairs):
        if not active[i]:
            continue
        f = filters.get(pair)
        if f is None:
            continue

        step = float(f.step_size)
        if step > 0:
            quantity[i] = _truncate_to_step(quantity[i], step)

        tick = float(f.tick_size)
        if tick > 0:
            entry_price[i] = _round_to_tick(entry_price[i], tick)
            stop_price[i] = _round_to_tick(stop_price[i], tick)
            take_profit_price[i] = _round_to_tick(take_profit_price[i], tick)

        min_qty = float(f.min_qty)
        if quantity[i] < min_qty:
            quantity[i] = 0.0
            notional[i] = 0.0
            active[i] = False
            continue

        max_qty = float(f.max_qty) if f.max_qty is not None else None
        if max_qty is not None and quantity[i] > max_qty:
            quantity[i] = 0.0
            notional[i] = 0.0
            active[i] = False
            continue

        min_notional = float(f.min_notional)
        row_notional = quantity[i] * entry_price[i]
        if row_notional < min_notional:
            quantity[i] = 0.0
            notional[i] = 0.0
            active[i] = False
            continue

        notional[i] = quantity[i] * entry_price[i]

    return BatchExecutionIntent(
        pairs=intents.pairs,
        action=intents.action,
        quantity=quantity,
        notional_usd=notional,
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        strategy_name=intents.strategy_name,
        strategy_variant=intents.strategy_variant,
        active_mask=active.astype(np.bool_),
    )

"""Semantic numeric types and converter helpers for domain boundaries.

Type aliases encode intent (Price vs Money vs Quantity) while remaining
plain ``Decimal`` at runtime.  Converter helpers are the canonical way to
construct these values — aliases themselves are **not** callable.

Kernel boundary utilities convert between ``Decimal`` sequences and numpy
``float64`` arrays for vectorized computation.
"""

import math
from collections.abc import Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dojiwick.domain.models.value_objects.candle import Candle

# Semantic type aliases (PEP 695)
type Price = Decimal
"""Asset prices: entry, stop, TP, close, liquidation, OHLC."""

type Money = Decimal
"""USD-denominated values: PnL, fees, equity, notional, funding."""

type Quantity = Decimal
"""Asset quantities, position sizes, volumes."""

"""Leverage ratios / multipliers."""

type Confidence = float
"""Confidence scores — analytics exception, stays float."""

type Rate = float
"""Analytics-only percentages: win rate, sharpe, etc."""

ZERO = Decimal(0)
"""Canonical zero Decimal for comparisons and arithmetic."""


# Converter helpers (runtime construction)
def _to_decimal(value: str | float | int | Decimal) -> Decimal:
    """Shared conversion logic with bool/NaN/Inf guards."""
    if isinstance(value, bool):
        raise TypeError(f"bool is not a valid numeric input: {value!r}")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is not a valid numeric value")
        if math.isinf(value):
            raise ValueError("Inf is not a valid numeric value")
        return Decimal(str(value))
    return Decimal(value)


def to_price(value: str | float | int | Decimal) -> Price:
    """Convert to Price (Decimal)."""
    return _to_decimal(value)


def to_money(value: str | float | int | Decimal) -> Money:
    """Convert to Money (Decimal)."""
    return _to_decimal(value)


def to_quantity(value: str | float | int | Decimal) -> Quantity:
    """Convert to Quantity (Decimal)."""
    return _to_decimal(value)


def decimals_to_array(values: Sequence[Decimal]) -> NDArray[np.float64]:
    """Convert a sequence of Decimals to a float64 numpy array."""
    return np.array([float(v) for v in values], dtype=np.float64)


def candles_to_ohlc(
    candles: Sequence[Candle],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extract (close, high, low) as float64 arrays from a candle sequence.

    Three separate passes for readability; negligible cost at ~60 candles.
    """
    close = decimals_to_array([c.close for c in candles])
    high = decimals_to_array([c.high for c in candles])
    low = decimals_to_array([c.low for c in candles])
    return close, high, low


# Exchange filter quantization


def quantize_qty_to_step(qty: Quantity, step_size: Quantity) -> Quantity:
    """Floor quantity to the nearest step multiple — exchanges reject finer precision."""
    if step_size <= 0:
        return qty
    return (qty // step_size) * step_size


def round_price_to_tick(price: Price, tick_size: Price, *, away_from: Price | None = None) -> Price:
    """Round price to the tick grid.

    With *away_from* set (protective stops), rounds away from that price so
    the rounded trigger never sits closer to the market than intended.
    """
    if tick_size <= 0:
        return price
    if away_from is None:
        return ((price / tick_size).to_integral_value(rounding=ROUND_HALF_UP)) * tick_size
    if price >= away_from:
        return ((price / tick_size).to_integral_value(rounding=ROUND_CEILING)) * tick_size
    return ((price / tick_size).to_integral_value(rounding=ROUND_FLOOR)) * tick_size


def meets_min_notional(qty: Quantity, price: Price, min_notional: Money) -> bool:
    """True when qty x price satisfies the exchange's minimum notional filter."""
    return qty * price >= min_notional

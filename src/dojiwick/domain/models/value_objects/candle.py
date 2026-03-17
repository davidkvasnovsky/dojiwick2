"""Candle domain model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.numerics import Price, Quantity
from dojiwick.domain.type_aliases import CandleInterval


@dataclass(slots=True, frozen=True, kw_only=True)
class Candle:
    """An OHLCV candle."""

    pair: str
    interval: CandleInterval
    open_time: datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    quote_volume: Quantity = Decimal(0)

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must not be empty")
        if not self.interval:
            raise ValueError("interval must not be empty")
        if self.open_time.tzinfo is None:
            raise ValueError("open_time must be timezone-aware")
        if self.open <= 0:
            raise ValueError("open must be positive")
        if self.high <= 0:
            raise ValueError("high must be positive")
        if self.low <= 0:
            raise ValueError("low must be positive")
        if self.close <= 0:
            raise ValueError("close must be positive")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")
        if self.quote_volume < 0:
            raise ValueError("quote_volume must be non-negative")

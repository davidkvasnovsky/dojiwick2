"""Settled perpetual funding rate event."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(slots=True, frozen=True, kw_only=True)
class FundingRate:
    """One settled funding event for a perpetual symbol. Immutable exchange fact."""

    symbol: str
    funding_time: datetime
    rate: Decimal

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.funding_time.tzinfo is None:
            raise ValueError("funding_time must be timezone-aware")
        if not Decimal(-1) < self.rate < Decimal(1):
            raise ValueError(f"rate must be in (-1, 1), got {self.rate}")

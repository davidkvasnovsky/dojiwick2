"""Settled perpetual funding rate event."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

# Perpetual funding settles at most every 8h (some symbols settle every 4h,
# which only yields more rows). This is the coverage tolerance for gap-fill
# and the max expected spacing between consecutive events.
MAX_FUNDING_INTERVAL = timedelta(hours=8)


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

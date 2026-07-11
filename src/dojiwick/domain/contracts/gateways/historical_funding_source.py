"""Historical funding source gateway protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.funding_rate import FundingRate


class HistoricalFundingSourcePort(Protocol):
    """Protocol for fetching settled funding events within a time range."""

    async def fetch_funding_range(self, symbol: str, start: datetime, end: datetime) -> tuple[FundingRate, ...]: ...

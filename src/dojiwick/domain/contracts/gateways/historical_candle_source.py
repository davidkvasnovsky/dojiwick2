"""Historical candle source gateway protocol for range-based candle fetching."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval


class HistoricalCandleSourcePort(Protocol):
    """Protocol for fetching historical candles within a time range."""

    async def fetch_candles_range(
        self, symbol: str, interval: CandleInterval, start: datetime, end: datetime
    ) -> tuple[Candle, ...]: ...

"""DB-first candle fetcher with exchange fallback.

Queries the database first via CandleRepositoryPort, fetches missing
candles from the exchange, and upserts them back to the DB.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.contracts.gateways.historical_candle_source import HistoricalCandleSourcePort
from dojiwick.domain.contracts.repositories.candle import CandleRepositoryPort
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CachingCandleFetcher:
    """Wraps a market data fetcher with DB-backed caching."""

    candle_repo: CandleRepositoryPort
    fetcher: HistoricalCandleSourcePort
    venue: str
    product: str

    async def fetch_candles_range(
        self,
        symbol: str,
        interval: CandleInterval,
        start: datetime,
        end: datetime,
    ) -> tuple[Candle, ...]:
        """Return candles for the range, using DB cache where available."""
        cached = await self.candle_repo.get_candles(
            symbol, interval, start, end, venue=self.venue, product=self.product
        )
        if cached:
            log.info("cache hit: %d candles for %s", len(cached), symbol)
            return cached

        log.info("cache miss for %s, fetching from exchange", symbol)
        fresh = await self.fetcher.fetch_candles_range(symbol, interval, start, end)
        if fresh:
            count = await self.candle_repo.upsert_candles(
                symbol, interval, fresh, venue=self.venue, product=self.product
            )
            log.info("cached %d candles for %s", count, symbol)
        return fresh

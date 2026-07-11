"""DB-first candle fetcher with exchange fallback.

Queries the database first via CandleRepositoryPort, fetches missing
candles from the exchange, and upserts them back to the DB.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from dojiwick.domain.contracts.gateways.historical_candle_source import HistoricalCandleSourcePort
from dojiwick.domain.contracts.repositories.candle import CandleRepositoryPort
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.timebase import interval_to_seconds
from dojiwick.domain.type_aliases import CandleInterval

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CachingCandleFetcher:
    """Wraps a market data fetcher with DB-backed caching.

    A cached slice that only partially overlaps the requested range must not
    be returned as-is — it would silently truncate the series. Instead the
    missing head/tail gaps are fetched from the exchange and merged. A pair
    listed after the requested start (rolling-joined universes) yields an
    empty head fetch and the cache remains authoritative. Interior gaps are
    the exchange's own (maintenance windows) and are not refetched.
    """

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
        if not cached:
            log.info("cache miss for %s, fetching from exchange", symbol)
            fresh = await self.fetcher.fetch_candles_range(symbol, interval, start, end)
            await self._store(symbol, interval, fresh)
            return fresh

        step = timedelta(seconds=interval_to_seconds(interval))
        head: tuple[Candle, ...] = ()
        tail: tuple[Candle, ...] = ()
        if cached[0].open_time > start + step:
            head = await self.fetcher.fetch_candles_range(symbol, interval, start, cached[0].open_time - step)
            await self._store(symbol, interval, head)
        if cached[-1].open_time < end - step:
            tail = await self.fetcher.fetch_candles_range(symbol, interval, cached[-1].open_time + step, end)
            await self._store(symbol, interval, tail)

        if not head and not tail:
            log.info("cache hit: %d candles for %s", len(cached), symbol)
            return cached

        log.info(
            "cache gap-fill for %s: %d cached + %d head + %d tail",
            symbol,
            len(cached),
            len(head),
            len(tail),
        )
        merged: dict[datetime, Candle] = {c.open_time: c for c in (*head, *cached, *tail)}
        return tuple(merged[t] for t in sorted(merged))

    async def _store(self, symbol: str, interval: CandleInterval, candles: tuple[Candle, ...]) -> None:
        if candles:
            count = await self.candle_repo.upsert_candles(
                symbol, interval, candles, venue=self.venue, product=self.product
            )
            log.info("cached %d candles for %s", count, symbol)

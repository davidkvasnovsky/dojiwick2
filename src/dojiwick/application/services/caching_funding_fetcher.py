"""Funding fetcher with DB-backed caching.

Mirrors CachingCandleFetcher: checks the funding_rates table first, fetches
missing head/tail gaps from the exchange, and upserts them back. Perpetual
funding settles at most every 8h (some symbols settle every 4h, which only
yields more rows), so coverage tolerance is fixed at 8h.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from dojiwick.domain.contracts.gateways.historical_funding_source import HistoricalFundingSourcePort
from dojiwick.domain.contracts.repositories.funding_rate import FundingRateRepositoryPort
from dojiwick.domain.models.value_objects.funding_rate import MAX_FUNDING_INTERVAL, FundingRate

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CachingFundingRateFetcher:
    """Wraps a funding source with DB-backed caching."""

    funding_repo: FundingRateRepositoryPort
    fetcher: HistoricalFundingSourcePort
    venue: str
    product: str

    async def fetch_funding_range(self, symbol: str, start: datetime, end: datetime) -> tuple[FundingRate, ...]:
        cached = await self.funding_repo.get_rates(symbol, start, end, venue=self.venue, product=self.product)
        if not cached:
            fresh = await self.fetcher.fetch_funding_range(symbol, start, end)
            await self._store(symbol, fresh)
            return fresh

        head: tuple[FundingRate, ...] = ()
        tail: tuple[FundingRate, ...] = ()
        if cached[0].funding_time > start + MAX_FUNDING_INTERVAL:
            head = await self.fetcher.fetch_funding_range(symbol, start, cached[0].funding_time - timedelta(seconds=1))
            await self._store(symbol, head)
        if cached[-1].funding_time < end - MAX_FUNDING_INTERVAL:
            tail = await self.fetcher.fetch_funding_range(symbol, cached[-1].funding_time + timedelta(seconds=1), end)
            await self._store(symbol, tail)

        if not head and not tail:
            return cached

        log.info("funding gap-fill for %s: %d cached + %d head + %d tail", symbol, len(cached), len(head), len(tail))
        merged: dict[datetime, FundingRate] = {r.funding_time: r for r in (*head, *cached, *tail)}
        return tuple(merged[t] for t in sorted(merged))

    async def _store(self, symbol: str, rates: tuple[FundingRate, ...]) -> None:
        if rates:
            count = await self.funding_repo.upsert_rates(symbol, rates, venue=self.venue, product=self.product)
            log.info("cached %d funding rates for %s", count, symbol)

"""Tests for caching candle fetcher."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from dojiwick.application.services.caching_candle_fetcher import CachingCandleFetcher
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval
from fixtures.fakes.candle_repository import InMemoryCandleRepo

_INTERVAL = CandleInterval("1h")
_VENUE = "binance"
_PRODUCT = "usd_c"


def _make_candle(pair: str, t: datetime) -> Candle:
    return Candle(
        pair=pair,
        interval=_INTERVAL,
        open_time=t,
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("95"),
        close=Decimal("102"),
        volume=Decimal("1000"),
    )


def _hourly(pair: str, start: datetime, count: int) -> tuple[Candle, ...]:
    return tuple(_make_candle(pair, start + timedelta(hours=i)) for i in range(count))


class FakeFetcher:
    """Range-aware fake exchange: serves only bars it has within [start, end]."""

    def __init__(self, candles: tuple[Candle, ...]) -> None:
        self.candles = candles
        self.calls: list[tuple[datetime, datetime]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def fetch_candles_range(
        self, symbol: str, interval: CandleInterval, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        del symbol, interval
        self.calls.append((start, end))
        return tuple(c for c in self.candles if start <= c.open_time <= end)


async def test_cache_miss_fetches_and_stores() -> None:
    repo = InMemoryCandleRepo()
    t = datetime(2025, 1, 1, tzinfo=UTC)
    candle = _make_candle("BTC/USDC", t)
    fetcher = FakeFetcher((candle,))
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("BTC/USDC", _INTERVAL, t, t + timedelta(hours=1))
    assert len(result) == 1
    assert fetcher.call_count == 1
    stored = await repo.get_candles("BTC/USDC", _INTERVAL, t, t + timedelta(hours=1), venue=_VENUE, product=_PRODUCT)
    assert len(stored) == 1


async def test_cache_hit_skips_fetch() -> None:
    repo = InMemoryCandleRepo()
    t = datetime(2025, 1, 1, tzinfo=UTC)
    candle = _make_candle("BTC/USDC", t)
    await repo.upsert_candles("BTC/USDC", _INTERVAL, (candle,), venue=_VENUE, product=_PRODUCT)

    fetcher = FakeFetcher(())
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("BTC/USDC", _INTERVAL, t, t + timedelta(hours=1))
    assert len(result) == 1
    assert fetcher.call_count == 0


async def test_full_coverage_multi_candle_hit() -> None:
    repo = InMemoryCandleRepo()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 2, tzinfo=UTC)
    await repo.upsert_candles("BTC/USDC", _INTERVAL, _hourly("BTC/USDC", start, 25), venue=_VENUE, product=_PRODUCT)

    fetcher = FakeFetcher(())
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("BTC/USDC", _INTERVAL, start, end)
    assert fetcher.call_count == 0
    assert len(result) == 25


async def test_partial_mid_slice_fetches_gaps_and_returns_full_range() -> None:
    """A cached slice inside the requested range must not truncate the series."""
    repo = InMemoryCandleRepo()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=9)
    exchange = _hourly("BTC/USDC", start, 9 * 24 + 1)
    # Cache only a 24h slice in the middle.
    mid = start + timedelta(days=4)
    await repo.upsert_candles("BTC/USDC", _INTERVAL, _hourly("BTC/USDC", mid, 24), venue=_VENUE, product=_PRODUCT)

    fetcher = FakeFetcher(exchange)
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("BTC/USDC", _INTERVAL, start, end)
    assert fetcher.call_count == 2  # head gap + tail gap
    assert len(result) == 9 * 24 + 1
    assert [c.open_time for c in result] == sorted(c.open_time for c in result)
    # Gaps are persisted back to the cache.
    stored = await repo.get_candles("BTC/USDC", _INTERVAL, start, end, venue=_VENUE, product=_PRODUCT)
    assert len(stored) == 9 * 24 + 1


async def test_cache_missing_tail_fetches_tail_only() -> None:
    repo = InMemoryCandleRepo()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    exchange = _hourly("BTC/USDC", start, 2 * 24 + 1)
    await repo.upsert_candles("BTC/USDC", _INTERVAL, _hourly("BTC/USDC", start, 24), venue=_VENUE, product=_PRODUCT)

    fetcher = FakeFetcher(exchange)
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("BTC/USDC", _INTERVAL, start, end)
    assert fetcher.call_count == 1  # tail gap only
    assert len(result) == 2 * 24 + 1


async def test_late_listing_pair_not_refetched() -> None:
    """Pair listed after requested start: cache is authoritative, head fetch is empty.

    Regression: a rolling-joined universe requests 2019.. for a pair listed in
    2020 — the complete cache must be a hit (plus one cheap empty head probe),
    never a full-range refetch.
    """
    repo = InMemoryCandleRepo()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    listing = start + timedelta(days=5)
    end = listing + timedelta(days=1)
    listed_bars = _hourly("DOGE/USDC", listing, 25)
    await repo.upsert_candles("DOGE/USDC", _INTERVAL, listed_bars, venue=_VENUE, product=_PRODUCT)

    fetcher = FakeFetcher(listed_bars)  # exchange has nothing before listing
    cacher = CachingCandleFetcher(candle_repo=repo, fetcher=fetcher, venue=_VENUE, product=_PRODUCT)

    result = await cacher.fetch_candles_range("DOGE/USDC", _INTERVAL, start, end)
    assert len(result) == 25
    assert result == listed_bars
    # Exactly one probe for the head gap, which returned empty — no full refetch.
    assert fetcher.calls == [(start, listing - timedelta(hours=1))]

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


class FakeFetcher:
    def __init__(self, candles: tuple[Candle, ...]) -> None:
        self.candles = candles
        self.call_count = 0

    async def fetch_candles_range(
        self, symbol: str, interval: CandleInterval, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        del symbol, interval, start, end
        self.call_count += 1
        return self.candles


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

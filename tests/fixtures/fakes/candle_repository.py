"""Candle repository test doubles."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval


@dataclass(slots=True)
class InMemoryCandleRepo:
    """In-memory candle store for tests."""

    _candles: list[Candle] = field(default_factory=list)

    async def upsert_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        candles: tuple[Candle, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        del symbol, interval, venue, product
        self._candles.extend(candles)
        return len(candles)

    async def get_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        start: datetime,
        end: datetime,
        *,
        venue: str,
        product: str,
    ) -> tuple[Candle, ...]:
        del venue, product
        return tuple(
            c for c in self._candles if c.pair == symbol and c.interval == interval and start <= c.open_time <= end
        )


class FailingCandleRepo:
    """Raises on all operations."""

    async def upsert_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        candles: tuple[Candle, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        del symbol, interval, candles, venue, product
        raise RuntimeError("candle repo failure")

    async def get_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        start: datetime,
        end: datetime,
        *,
        venue: str,
        product: str,
    ) -> tuple[Candle, ...]:
        del symbol, interval, start, end, venue, product
        raise RuntimeError("candle repo failure")

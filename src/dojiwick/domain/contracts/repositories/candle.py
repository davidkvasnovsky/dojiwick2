"""Candle repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval


class CandleRepositoryPort(Protocol):
    """OHLCV data persistence."""

    async def upsert_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        candles: tuple[Candle, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        """Upsert candle rows and return the number upserted."""
        ...

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
        """Return candles for the given symbol/interval/time range, scoped by venue/product."""
        ...

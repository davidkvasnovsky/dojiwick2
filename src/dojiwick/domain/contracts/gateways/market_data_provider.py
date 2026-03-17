"""Market data provider gateway protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.numerics import Price
from dojiwick.domain.type_aliases import CandleInterval


class MarketDataProviderPort(Protocol):
    """External market data source for prices and candles."""

    async def fetch_latest_prices(self, pairs: tuple[str, ...]) -> dict[str, Price]:
        """Return the latest price for each requested pair."""
        ...

    async def fetch_candles(self, pair: str, interval: CandleInterval, limit: int) -> tuple[Candle, ...]:
        """Return the most recent candles for a pair/interval."""
        ...

    async def ping(self) -> bool:
        """Return True if the data source is reachable."""
        ...

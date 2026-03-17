"""Market data provider test doubles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.numerics import Price, to_price
from dojiwick.domain.type_aliases import CandleInterval


@dataclass(slots=True)
class InMemoryMarketDataProvider:
    """In-memory market data source for tests."""

    _prices: dict[str, Price] = field(default_factory=dict)
    _candles: dict[str, list[Candle]] = field(default_factory=dict)

    async def fetch_latest_prices(self, pairs: tuple[str, ...]) -> dict[str, Price]:
        """Return the latest price for each requested pair."""
        return {p: self._prices[p] for p in pairs if p in self._prices}

    async def fetch_candles(self, pair: str, interval: CandleInterval, limit: int) -> tuple[Candle, ...]:
        """Return the most recent candles for a pair/interval."""
        key = f"{pair}:{interval}"
        stored = self._candles.get(key, [])
        return tuple(stored[-limit:])

    async def ping(self) -> bool:
        """Return True (always reachable in memory)."""
        return True

    async def fetch_candles_range(
        self, pair: str, interval: CandleInterval, start: datetime, end: datetime
    ) -> tuple[Candle, ...]:
        """Return candles within a time range for a pair/interval."""
        key = f"{pair}:{interval}"
        stored = self._candles.get(key, [])
        return tuple(c for c in stored if start <= c.open_time <= end)

    def set_prices(self, prices: Mapping[str, float | Price]) -> None:
        """Test helper: set the price map."""
        normalized = {symbol: to_price(price) for symbol, price in prices.items()}
        self._prices.update(normalized)

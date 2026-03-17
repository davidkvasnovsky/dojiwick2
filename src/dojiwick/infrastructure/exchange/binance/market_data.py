"""Binance market data provider — prices, candles, and connectivity checks."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.numerics import Price
from dojiwick.domain.type_aliases import CandleInterval

from .boundary import parse_price, parse_quantity
from .http_client import BinanceHttpClient

log = logging.getLogger(__name__)

_MAX_KLINES_PER_REQUEST = 1500


def _parse_kline_entry(pair: str, interval: CandleInterval, arr: list[object]) -> Candle | None:
    """Parse a single Binance kline array into a Candle, or None on bad data."""
    if len(arr) < 12:
        return None
    open_time_ms = arr[0]
    if not isinstance(open_time_ms, int | float):
        return None
    return Candle(
        pair=pair,
        interval=interval,
        open_time=datetime.fromtimestamp(int(open_time_ms) / 1000, tz=UTC),
        open=parse_price(str(arr[1])),
        high=parse_price(str(arr[2])),
        low=parse_price(str(arr[3])),
        close=parse_price(str(arr[4])),
        volume=parse_quantity(str(arr[5])),
        quote_volume=parse_quantity(str(arr[7])),
    )


def _parse_kline_list(pair: str, interval: CandleInterval, raw: list[object]) -> tuple[Candle, ...]:
    """Parse a full Binance klines response into a tuple of Candles."""
    candles: list[Candle] = []
    for entry in raw:
        if not isinstance(entry, list):
            continue
        candle = _parse_kline_entry(pair, interval, cast(list[object], entry))
        if candle is not None:
            candles.append(candle)
    return tuple(candles)


@dataclass(slots=True)
class BinanceMarketDataProvider:
    """Fetches market data from the Binance Futures REST API."""

    client: BinanceHttpClient

    async def ping(self) -> bool:
        """Return True if the Binance API is reachable."""
        try:
            await self.client.request("GET", "/fapi/v1/ping")
            return True
        except Exception:
            log.warning("ping failed", exc_info=True)
            return False

    async def fetch_latest_prices(self, pairs: tuple[str, ...]) -> dict[str, Price]:
        """Return the latest price for each requested pair (symbol format)."""
        raw = await self.client.request_list("GET", "/fapi/v2/ticker/price")
        pair_set = set(pairs)
        result: dict[str, Price] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            item = cast(dict[str, object], entry)
            symbol = item.get("symbol")
            price_str = item.get("price")
            if isinstance(symbol, str) and isinstance(price_str, str) and symbol in pair_set:
                result[symbol] = parse_price(price_str)
        return result

    async def fetch_candles(self, pair: str, interval: CandleInterval, limit: int) -> tuple[Candle, ...]:
        """Return the most recent candles for a pair/interval."""
        raw = await self.client.request_list(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": pair, "interval": interval, "limit": str(limit)},
        )
        return _parse_kline_list(pair, interval, raw)

    async def fetch_candles_range(
        self,
        symbol: str,
        interval: CandleInterval,
        start: datetime,
        end: datetime,
    ) -> tuple[Candle, ...]:
        """Fetch candles for a date range with automatic pagination.

        Not on ``MarketDataProviderPort`` — concrete method for backtest CLI only.
        """
        candles: list[Candle] = []
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while start_ms < end_ms:
            raw = await self.client.request_list(
                "GET",
                "/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": str(start_ms),
                    "endTime": str(end_ms),
                    "limit": str(_MAX_KLINES_PER_REQUEST),
                },
            )
            batch = _parse_kline_list(symbol, interval, raw)

            if not batch:
                break
            candles.extend(batch)
            last_open_ms = int(batch[-1].open_time.timestamp() * 1000)
            start_ms = last_open_ms + 1
            if len(batch) < _MAX_KLINES_PER_REQUEST:
                break

        return tuple(candles)

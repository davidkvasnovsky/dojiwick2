"""Unit tests for BinanceMarketDataProvider."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from dojiwick.domain.type_aliases import CandleInterval
from dojiwick.infrastructure.exchange.binance.market_data import BinanceMarketDataProvider


def _make_provider() -> tuple[BinanceMarketDataProvider, MagicMock]:
    mock_client = MagicMock()
    mock_client.request = AsyncMock()
    mock_client.request_list = AsyncMock()
    provider = BinanceMarketDataProvider.__new__(BinanceMarketDataProvider)
    object.__setattr__(provider, "client", mock_client)
    return provider, mock_client


class TestPing:
    async def test_ping_success(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value={})
        assert await provider.ping() is True

    async def test_ping_failure(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(side_effect=Exception("boom"))
        assert await provider.ping() is False


class TestFetchLatestPrices:
    async def test_parses_prices(self) -> None:
        provider, client = _make_provider()
        client.request_list = AsyncMock(
            return_value=[
                {"symbol": "BTCUSDT", "price": "50000.00"},
                {"symbol": "ETHUSDT", "price": "3000.00"},
                {"symbol": "SOLUSDT", "price": "100.00"},
            ]
        )
        result = await provider.fetch_latest_prices(("BTCUSDT", "ETHUSDT"))
        assert result == {
            "BTCUSDT": Decimal("50000.00"),
            "ETHUSDT": Decimal("3000.00"),
        }
        assert "SOLUSDT" not in result

    async def test_skips_malformed_entries(self) -> None:
        provider, client = _make_provider()
        client.request_list = AsyncMock(
            return_value=[
                {"symbol": "BTCUSDT", "price": "50000.00"},
                "not a dict",
                {"symbol": 123, "price": "bad"},
            ]
        )
        result = await provider.fetch_latest_prices(("BTCUSDT",))
        assert len(result) == 1


class TestFetchCandles:
    async def test_parses_candles(self) -> None:
        provider, client = _make_provider()
        raw_kline = [
            1704067200000,
            "42000.00",
            "43000.00",
            "41000.00",
            "42500.00",
            "100.5",
            1704153599999,
            "4200000.00",
            200,
            "50.0",
            "2100000.00",
            "0",
        ]
        client.request_list = AsyncMock(return_value=[raw_kline])
        candles = await provider.fetch_candles("BTCUSDT", CandleInterval("1h"), 1)
        assert len(candles) == 1
        c = candles[0]
        assert c.pair == "BTCUSDT"
        assert c.interval == "1h"
        assert c.open_time == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        assert c.open == Decimal("42000.00")
        assert c.high == Decimal("43000.00")
        assert c.low == Decimal("41000.00")
        assert c.close == Decimal("42500.00")
        assert c.volume == Decimal("100.5")
        assert c.quote_volume == Decimal("4200000.00")

    async def test_skips_short_arrays(self) -> None:
        provider, client = _make_provider()
        client.request_list = AsyncMock(return_value=[[1, 2, 3]])
        candles = await provider.fetch_candles("BTCUSDT", CandleInterval("1h"), 1)
        assert len(candles) == 0

"""Unit tests for BinanceExchangeMetadataProvider."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from dojiwick.domain.errors import ExchangeError
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.infrastructure.exchange.binance.exchange_metadata import BinanceExchangeMetadataProvider


def _make_provider() -> tuple[BinanceExchangeMetadataProvider, MagicMock]:
    mock_client = MagicMock()
    mock_client.request = AsyncMock()
    provider = BinanceExchangeMetadataProvider.__new__(BinanceExchangeMetadataProvider)
    object.__setattr__(provider, "client", mock_client)
    object.__setattr__(provider, "_exchange_info", None)
    return provider, mock_client


_EXCHANGE_INFO: dict[str, object] = {
    "symbols": [
        {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "baseAssetPrecision": 8,
            "quotePrecision": 8,
            "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000.00", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000.00", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "10.0"},
            ],
        },
        {
            "symbol": "ETHUSDT",
            "status": "BREAK",
            "baseAsset": "ETH",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 3,
            "baseAssetPrecision": 8,
            "quotePrecision": 8,
            "filters": [],
        },
    ]
}


class TestListInstruments:
    async def test_filters_trading_only(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        instruments = await provider.list_instruments(BINANCE_VENUE, BINANCE_USD_C)
        assert len(instruments) == 1
        inst = instruments[0]
        assert inst.instrument_id.symbol == "BTCUSDT"
        assert inst.status == "TRADING"

    async def test_instrument_filters_parsed(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        instruments = await provider.list_instruments(BINANCE_VENUE, BINANCE_USD_C)
        f = instruments[0].filters
        assert f.min_price == Decimal("0.01")
        assert f.max_price == Decimal("100000.00")
        assert f.tick_size == Decimal("0.01")
        assert f.min_qty == Decimal("0.001")
        assert f.max_qty == Decimal("1000.00")
        assert f.step_size == Decimal("0.001")
        assert f.min_notional == Decimal("10.0")

    async def test_precision_parsed(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        instruments = await provider.list_instruments(BINANCE_VENUE, BINANCE_USD_C)
        inst = instruments[0]
        assert inst.price_precision == 2
        assert inst.quantity_precision == 3
        assert inst.base_asset_precision == 8
        assert inst.quote_asset_precision == 8


class TestGetInstrument:
    async def test_found(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        iid = InstrumentId(
            venue=BINANCE_VENUE,
            product=BINANCE_USD_C,
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            settle_asset="USDT",
        )
        inst = await provider.get_instrument(iid)
        assert inst.instrument_id.symbol == "BTCUSDT"

    async def test_not_found_raises(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        iid = InstrumentId(
            venue=BINANCE_VENUE,
            product=BINANCE_USD_C,
            symbol="XYZUSDT",
            base_asset="XYZ",
            quote_asset="USDT",
            settle_asset="USDT",
        )
        with pytest.raises(ExchangeError, match="instrument not found"):
            await provider.get_instrument(iid)


class TestGetCapabilities:
    async def test_returns_binance_defaults(self) -> None:
        provider, _ = _make_provider()

        caps = await provider.get_capabilities(BINANCE_VENUE, BINANCE_USD_C)
        assert caps.venue == BINANCE_VENUE
        assert caps.product == BINANCE_USD_C
        assert caps.supports_hedge_mode is True
        assert caps.supports_stop_market is True
        assert caps.supports_take_profit_market is True
        assert caps.max_open_orders == 200
        assert "GTC" in caps.supported_time_in_force


class TestExchangeInfoCaching:
    async def test_cached_no_refetch(self) -> None:
        provider, client = _make_provider()
        client.request = AsyncMock(return_value=_EXCHANGE_INFO)

        await provider.list_instruments(BINANCE_VENUE, BINANCE_USD_C)
        await provider.list_instruments(BINANCE_VENUE, BINANCE_USD_C)

        assert client.request.call_count == 1

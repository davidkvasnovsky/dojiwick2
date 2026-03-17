"""Unit tests for BinanceAccountStateProvider."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.errors import ExchangeError
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.infrastructure.exchange.binance.account_state import BinanceAccountStateProvider


_EXCHANGE_INFO_RESPONSE: dict[str, object] = {
    "symbols": [
        {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
        {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT"},
    ],
}


def _make_provider() -> tuple[BinanceAccountStateProvider, MagicMock]:
    mock_client = MagicMock()

    async def _mock_request(method: str, path: str, **kwargs: object) -> object:
        if path == "/fapi/v1/exchangeInfo":
            return _EXCHANGE_INFO_RESPONSE
        if path == "/fapi/v2/account":
            return _ACCOUNT_RESPONSE
        raise ValueError(f"unexpected request: {method} {path}")

    mock_client.request = AsyncMock(side_effect=_mock_request)
    provider = BinanceAccountStateProvider.__new__(BinanceAccountStateProvider)
    object.__setattr__(provider, "client", mock_client)
    object.__setattr__(provider, "_symbol_assets", None)
    return provider, mock_client


_ACCOUNT_RESPONSE: dict[str, object] = {
    "totalWalletBalance": "10000.00",
    "availableBalance": "8000.00",
    "totalUnrealizedProfit": "500.00",
    "totalMarginBalance": "10500.00",
    "assets": [
        {
            "asset": "USDT",
            "walletBalance": "10000.00",
            "availableBalance": "8000.00",
            "crossUnPnl": "500.00",
        },
    ],
    "positions": [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.1",
            "positionSide": "BOTH",
            "entryPrice": "42000.00",
            "unrealizedProfit": "500.00",
            "leverage": 10,
        },
        {
            "symbol": "ETHUSDT",
            "positionAmt": "0",
            "positionSide": "BOTH",
            "entryPrice": "0",
            "unrealizedProfit": "0",
            "leverage": 1,
        },
    ],
}


class TestGetAccountSnapshot:
    async def test_full_snapshot_parsing(self) -> None:
        provider, _client = _make_provider()

        snap = await provider.get_account_snapshot("default")

        assert snap.account == "default"
        assert snap.total_wallet_balance == Decimal("10000.00")
        assert snap.available_balance == Decimal("8000.00")
        assert snap.total_unrealized_pnl == Decimal("500.00")
        assert snap.total_margin_balance == Decimal("10500.00")

        assert len(snap.balances) == 1
        assert snap.balances[0].asset == "USDT"
        assert snap.balances[0].wallet_balance == Decimal("10000.00")

    async def test_zero_positions_filtered(self) -> None:
        provider, _client = _make_provider()

        snap = await provider.get_account_snapshot("default")

        assert len(snap.positions) == 1
        pos = snap.positions[0]
        assert pos.instrument_id.symbol == "BTCUSDT"
        assert pos.quantity == Decimal("0.1")
        assert pos.entry_price == Decimal("42000.00")
        assert pos.position_side == PositionSide.NET
        assert pos.leverage == 10

    async def test_instrument_id_built_correctly(self) -> None:
        provider, _client = _make_provider()

        snap = await provider.get_account_snapshot("default")
        iid = snap.positions[0].instrument_id
        assert iid.venue == BINANCE_VENUE
        assert iid.product == BINANCE_USD_C
        assert iid.base_asset == "BTC"
        assert iid.quote_asset == "USDT"

    async def test_unknown_symbol_raises(self) -> None:
        """Symbol not in exchange info raises ExchangeError."""
        provider, _client = _make_provider()
        # Prime the cache with exchange info (no XYZUSDT)
        provider._symbol_assets = {"BTCUSDT": ("BTC", "USDT")}  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(ExchangeError, match="XYZUSDT.*not found"):
            await provider._resolve_assets("XYZUSDT")  # pyright: ignore[reportPrivateUsage]

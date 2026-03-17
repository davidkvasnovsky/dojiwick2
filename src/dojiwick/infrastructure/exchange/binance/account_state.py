"""Binance account-state adapter — balances and positions from REST API."""

from dataclasses import dataclass, field
from typing import cast

from dojiwick.domain.models.value_objects.account_state import (
    AccountBalance,
    AccountSnapshot,
    ExchangePositionLeg,
)
from dojiwick.domain.numerics import ZERO

from .boundary import build_instrument_id, parse_money, parse_position_side, parse_price, parse_quantity
from .http_client import BinanceHttpClient
from .response_types import AccountResponse


@dataclass(slots=True)
class BinanceAccountStateProvider:
    """Fetches account balances and open positions from Binance Futures."""

    client: BinanceHttpClient
    _symbol_assets: dict[str, tuple[str, str]] | None = field(default=None, init=False, repr=False)

    async def _resolve_assets(self, symbol: str) -> tuple[str, str]:
        """Resolve base/quote assets from exchange info (lazy init)."""
        if self._symbol_assets is None:
            from .response_types import ExchangeInfoResponse

            info = cast(ExchangeInfoResponse, await self.client.request("GET", "/fapi/v1/exchangeInfo"))
            self._symbol_assets = {sym["symbol"]: (sym["baseAsset"], sym["quoteAsset"]) for sym in info["symbols"]}
        assets = self._symbol_assets.get(symbol)
        if assets is not None:
            return assets
        from dojiwick.domain.errors import ExchangeError

        raise ExchangeError(f"symbol {symbol} not found in exchange info — cannot resolve base/quote assets")

    async def get_account_snapshot(self, account: str) -> AccountSnapshot:
        """Fetch full account snapshot via GET /fapi/v2/account (signed)."""
        raw = await self.client.request("GET", "/fapi/v2/account", signed=True)
        data = cast(AccountResponse, raw)

        balances: list[AccountBalance] = []
        for entry in data["assets"]:
            balances.append(
                AccountBalance(
                    asset=entry["asset"],
                    wallet_balance=parse_money(entry.get("walletBalance", "0")),
                    available_balance=parse_money(entry.get("availableBalance", "0")),
                    cross_unrealized_pnl=parse_money(entry.get("crossUnPnl", "0")),
                )
            )

        positions: list[ExchangePositionLeg] = []
        for entry in data["positions"]:
            qty = parse_quantity(entry.get("positionAmt", "0"))
            if qty == ZERO:
                continue
            symbol = entry["symbol"]
            if not symbol:
                continue
            leverage_raw = entry.get("leverage", 1)
            base_asset, quote_asset = await self._resolve_assets(symbol)
            positions.append(
                ExchangePositionLeg(
                    instrument_id=build_instrument_id(
                        symbol=symbol,
                        base_asset=base_asset,
                        quote_asset=quote_asset,
                    ),
                    position_side=parse_position_side(entry.get("positionSide", "BOTH")),
                    quantity=abs(qty),
                    entry_price=parse_price(entry.get("entryPrice", "0")),
                    unrealized_pnl=parse_money(entry.get("unrealizedProfit", "0")),
                    leverage=int(leverage_raw),
                )
            )

        return AccountSnapshot(
            account=account,
            balances=tuple(balances),
            positions=tuple(positions),
            total_wallet_balance=parse_money(data.get("totalWalletBalance", "0")),
            available_balance=parse_money(data.get("availableBalance", "0")),
            total_unrealized_pnl=parse_money(data.get("totalUnrealizedProfit", "0")),
            total_margin_balance=parse_money(data.get("totalMarginBalance", "0")),
        )

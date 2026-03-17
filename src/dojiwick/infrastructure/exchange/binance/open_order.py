"""Binance open order adapter — query and cancel open orders."""

import logging
from dataclasses import dataclass
from typing import cast

from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder

from .boundary import parse_order_side, parse_order_status, parse_position_side, parse_quantity, str_field
from .http_client import BinanceHttpClient

log = logging.getLogger(__name__)


def parse_open_order(raw: dict[str, object]) -> ExchangeOpenOrder:
    """Parse a single Binance open order response into domain type."""
    return ExchangeOpenOrder(
        exchange_order_id=str_field(raw, "orderId"),
        client_order_id=str_field(raw, "clientOrderId"),
        symbol=str_field(raw, "symbol"),
        side=parse_order_side(str_field(raw, "side")),
        position_side=parse_position_side(str_field(raw, "positionSide")),
        status=parse_order_status(str_field(raw, "status")),
        original_quantity=parse_quantity(str_field(raw, "origQty")),
        filled_quantity=parse_quantity(str_field(raw, "executedQty")),
    )


@dataclass(slots=True)
class BinanceOpenOrderAdapter:
    """Queries and cancels open orders on the Binance Futures API."""

    client: BinanceHttpClient

    async def get_open_orders(self, symbol: str) -> tuple[ExchangeOpenOrder, ...]:
        """Fetch all open orders for a symbol."""
        raw_list = await self.client.request_list("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)
        orders: list[ExchangeOpenOrder] = []
        for item in raw_list:
            if isinstance(item, dict):
                orders.append(parse_open_order(cast(dict[str, object], item)))
        return tuple(orders)

    async def cancel_all_open_orders(self, symbol: str) -> None:
        """Cancel all open orders for a symbol."""
        await self.client.request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)
        log.info("cancelled all open orders for %s", symbol)

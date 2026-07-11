"""Binance user-data WebSocket stream for order lifecycle events.

Manages the listenKey lifecycle (create, keepalive, delete) and streams
ORDER_TRADE_UPDATE events as ``ExchangeOrderUpdate`` or ``OrderEvent`` objects.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.order_event_stream import StreamCursor
from dojiwick.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate

from .boundary import ms_to_utc, parse_ws_order_update, str_field
from .http_client import BinanceHttpClient

if TYPE_CHECKING:
    import aiohttp

log = logging.getLogger(__name__)

_PROD_WS_BASE = "wss://fstream.binance.com/ws/"
_TESTNET_WS_BASE = "wss://stream.binancefuture.com/ws/"
_KEEPALIVE_INTERVAL_SEC = 30 * 60  # 30 minutes


@dataclass(slots=True)
class BinanceOrderEventStream:
    """WebSocket stream for Binance order events with listenKey lifecycle."""

    client: BinanceHttpClient
    clock: ClockPort
    keepalive_failure_threshold: int = 2

    _ws: aiohttp.ClientWebSocketResponse | None = field(default=None, init=False, repr=False)
    _listen_key: str = field(default="", init=False, repr=False)
    _keepalive_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _sequence: int = field(default=0, init=False)
    _connected: bool = field(default=False, init=False)

    @property
    def stream_name(self) -> str:
        return "binance_orders"

    async def connect(self) -> None:
        """POST listenKey, open WS, start keepalive."""
        data = await self.client.request("POST", "/fapi/v1/listenKey", signed=True)
        self._listen_key = str_field(data, "listenKey")
        if not self._listen_key:
            raise ConnectionError("failed to obtain listenKey")

        ws_base = _TESTNET_WS_BASE if self.client.testnet else _PROD_WS_BASE
        url = f"{ws_base}{self._listen_key}"

        session = await self.client.ensure_session()
        self._ws = await session.ws_connect(url)
        self._connected = True
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        log.info("WS user-data stream connected")

    async def _keepalive_loop(self) -> None:
        """PUT listenKey every 30 minutes to prevent expiry.

        After ``keepalive_failure_threshold`` consecutive failures the socket
        is force-closed: the listenKey expires at ~60 minutes, and a silently
        dead key means order events stop with no visible signal. Closing lets
        the consumer's supervisor reconnect with a fresh key.
        """
        failures = 0
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_SEC)
            try:
                await self.client.request("PUT", "/fapi/v1/listenKey", signed=True)
                failures = 0
                log.debug("listenKey keepalive sent")
            except Exception:
                failures += 1
                log.warning("listenKey keepalive failed (%d consecutive)", failures, exc_info=True)
                if failures >= self.keepalive_failure_threshold:
                    log.critical("listenKey keepalive failing — forcing reconnect with fresh key")
                    self._connected = False
                    if self._ws is not None:
                        await self._ws.close()
                    return

    async def disconnect(self) -> None:
        """Cancel keepalive, close WS, DELETE listenKey."""
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            self._keepalive_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        if self._listen_key:
            try:
                await self.client.request("DELETE", "/fapi/v1/listenKey", signed=True)
            except Exception:
                log.warning("listenKey DELETE failed", exc_info=True)
            self._listen_key = ""

        self._connected = False
        log.info("WS user-data stream disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def raw_updates(self) -> AsyncIterator[ExchangeOrderUpdate]:
        """Yield parsed ExchangeOrderUpdate objects from the WebSocket."""
        import aiohttp as _aiohttp

        if self._ws is None:
            return

        async for msg in self._ws:
            if msg.type == _aiohttp.WSMsgType.TEXT:
                raw = cast(dict[str, object], msg.json())
                event_type = str(raw.get("e", ""))
                if event_type != "ORDER_TRADE_UPDATE":
                    continue
                self._sequence += 1
                try:
                    update = parse_ws_order_update(raw)
                    yield update
                except Exception:
                    log.warning("failed to parse ORDER_TRADE_UPDATE", exc_info=True)
            elif msg.type in (
                _aiohttp.WSMsgType.CLOSED,
                _aiohttp.WSMsgType.ERROR,
            ):
                self._connected = False
                log.warning("WS connection closed/errored")
                break

    async def replay_trades(self, symbol: str, start_time_ms: int) -> tuple[ExchangeOrderUpdate, ...]:
        """REST recovery sweep for one symbol: userTrades since *start_time_ms*.

        Both /fapi/v1/userTrades and /fapi/v1/order are per-symbol-mandatory
        endpoints — the previous account-wide allOrders call was rejected by
        the exchange on every startup with a stored cursor.
        """
        params: dict[str, str] = {"symbol": symbol, "limit": "1000"}
        if start_time_ms > 0:
            params["startTime"] = str(start_time_ms)
        trades = await self.client.request_list("GET", "/fapi/v1/userTrades", params=params, signed=True)

        by_order: dict[int, list[dict[str, object]]] = {}
        for raw in trades:
            if not isinstance(raw, dict):
                continue
            item = cast(dict[str, object], raw)
            order_id = item.get("orderId")
            if isinstance(order_id, int):
                by_order.setdefault(order_id, []).append(item)

        updates: list[ExchangeOrderUpdate] = []
        for order_id, order_trades in by_order.items():
            order_raw = await self.client.request(
                "GET", "/fapi/v1/order", params={"symbol": symbol, "orderId": str(order_id)}, signed=True
            )
            status = OrderStatus(str_field(order_raw, "status").lower())
            client_order_id = str_field(order_raw, "clientOrderId")
            cumulative = Decimal(str(order_raw.get("executedQty", "0")))
            avg_price = Decimal(str(order_raw.get("avgPrice", "0")))
            side = OrderSide(str_field(order_raw, "side").lower())
            position_side = PositionSide(str_field(order_raw, "positionSide", "BOTH").replace("BOTH", "net").lower())
            reduce_only = bool(order_raw.get("reduceOnly", False))
            close_position = bool(order_raw.get("closePosition", False))

            for trade in sorted(order_trades, key=lambda t: int(str(t.get("time", 0)))):
                trade_time = ms_to_utc(int(str(trade.get("time", 0))))
                updates.append(
                    ExchangeOrderUpdate(
                        exchange_order_id=str(order_id),
                        client_order_id=client_order_id,
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        order_status=status,
                        execution_type="TRADE",
                        position_side=position_side,
                        last_filled_qty=Decimal(str(trade.get("qty", "0"))),
                        last_filled_price=Decimal(str(trade.get("price", "0"))),
                        cumulative_filled_qty=cumulative,
                        avg_price=avg_price,
                        commission=Decimal(str(trade.get("commission", "0"))),
                        commission_asset=str(trade.get("commissionAsset", "")),
                        trade_id=int(str(trade.get("id", 0))),
                        order_trade_time=trade_time,
                        reduce_only=reduce_only,
                        close_position=close_position,
                        realized_profit=Decimal(str(trade.get("realizedPnl", "0"))),
                        event_time=trade_time,
                        transaction_time=trade_time,
                    )
                )

        return tuple(updates)

    async def get_cursor(self) -> StreamCursor:
        """Return the current stream position as a cursor."""
        now_ms = int(self.clock.now_utc().timestamp() * 1000)
        return StreamCursor(stream_name=self.stream_name, sequence=self._sequence, timestamp_ms=now_ms)

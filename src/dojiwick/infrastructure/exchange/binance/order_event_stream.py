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
from dojiwick.domain.contracts.gateways.order_event_stream import StreamCursor, StreamGap
from dojiwick.domain.enums import OrderStatus, STATUS_TO_EVENT_TYPE
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_event import OrderEvent

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

    _ws: aiohttp.ClientWebSocketResponse | None = field(default=None, init=False, repr=False)
    _listen_key: str = field(default="", init=False)
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
        """PUT listenKey every 30 minutes to prevent expiry."""
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL_SEC)
            try:
                await self.client.request("PUT", "/fapi/v1/listenKey", signed=True)
                log.debug("listenKey keepalive sent")
            except Exception:
                log.warning("listenKey keepalive failed", exc_info=True)

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

    async def events(self) -> AsyncIterator[OrderEvent]:
        """Yield OrderEvent objects converted from ExchangeOrderUpdate."""
        async for update in self.raw_updates():
            event_type = STATUS_TO_EVENT_TYPE.get(update.order_status)
            if event_type is None:
                continue
            yield OrderEvent(
                order_id=0,
                event_type=event_type,
                occurred_at=update.event_time,
                exchange_order_id=update.exchange_order_id,
                filled_quantity=update.last_filled_qty,
                fees_usd=update.commission,
                fee_asset=update.commission_asset,
                native_fee_amount=update.commission,
            )

    async def replay_from(self, cursor: StreamCursor) -> AsyncIterator[OrderEvent]:
        """REST GET /fapi/v1/allOrders since cursor timestamp, convert to OrderEvent."""
        params: dict[str, str] = {}
        if cursor.timestamp_ms > 0:
            params["startTime"] = str(cursor.timestamp_ms)

        orders = await self.client.request_list("GET", "/fapi/v1/allOrders", params=params, signed=True)
        for order_raw in orders:
            if not isinstance(order_raw, dict):
                continue
            item = cast(dict[str, object], order_raw)
            status_str = str_field(item, "status")
            event_type = STATUS_TO_EVENT_TYPE.get(OrderStatus(status_str.lower()))
            if event_type is None:
                continue
            update_time = item.get("updateTime", 0)
            if isinstance(update_time, int | float):
                occurred_at = ms_to_utc(int(update_time))
            else:
                occurred_at = ms_to_utc(0)
            yield OrderEvent(
                order_id=0,
                event_type=event_type,
                occurred_at=occurred_at,
                exchange_order_id=str(item.get("orderId", "")),
                filled_quantity=Decimal(str(item.get("executedQty", "0"))),
                fees_usd=Decimal(0),
            )

    async def detect_gaps(self, since: StreamCursor) -> tuple[StreamGap, ...]:
        """Detect gaps — placeholder returning no gaps (sequence tracking is local)."""
        _ = since
        return ()

    async def recover_gap(self, gap: StreamGap) -> tuple[OrderEvent, ...]:
        """REST fetch for missing range."""
        cursor = StreamCursor(
            stream_name=self.stream_name, sequence=gap.start_sequence, timestamp_ms=gap.detected_at_ms
        )
        events: list[OrderEvent] = []
        async for event in self.replay_from(cursor):
            events.append(event)
        return tuple(events)

    async def get_cursor(self) -> StreamCursor:
        """Return the current stream position as a cursor."""
        now_ms = int(self.clock.now_utc().timestamp() * 1000)
        return StreamCursor(stream_name=self.stream_name, sequence=self._sequence, timestamp_ms=now_ms)

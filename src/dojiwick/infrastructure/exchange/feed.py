"""WebSocket-first data feed with REST bootstrap and fallback recovery.

The ``ExchangeDataFeed`` populates an ``ExchangeCache`` via WebSocket
streams. On startup it bootstraps from REST. When the WS connection drops
or gaps are detected, it falls back to REST automatically and re-populates
the cache until WS reconnects.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from dojiwick.domain.contracts.gateways.account_state import AccountStatePort
from dojiwick.domain.contracts.gateways.market_data_provider import MarketDataProviderPort
from dojiwick.domain.contracts.gateways.order_event_stream import (
    StreamCursor,
    StreamGap,
)
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.infrastructure.exchange.cache import ExchangeCache

log = logging.getLogger(__name__)


class FeedStatus(StrEnum):
    """Current state of the data feed."""

    DISCONNECTED = "disconnected"
    BOOTSTRAPPING = "bootstrapping"
    WS_ACTIVE = "ws_active"
    REST_FALLBACK = "rest_fallback"


class OrderStreamAdapter(Protocol):
    """Minimal order stream contract used by the data feed.

    This is a subset of OrderEventStreamPort covering only the methods
    the feed needs: connection lifecycle, gap detection, and recovery.
    """

    @property
    def stream_name(self) -> str: ...

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    async def detect_gaps(self, since: StreamCursor) -> tuple[StreamGap, ...]: ...
    async def recover_gap(self, gap: StreamGap) -> tuple[OrderEvent, ...]: ...
    async def get_cursor(self) -> StreamCursor: ...


@dataclass(slots=True)
class ExchangeDataFeed:
    """WS-first data feed with REST bootstrap and gap recovery.

    Lifecycle:
    1. ``bootstrap()`` — REST-fetch initial state into cache.
    2. ``start()`` — Attempt WS connect; on failure, activate REST polling.
    3. ``refresh_via_rest()`` — Manual or automatic REST refresh.
    4. ``stop()`` — Disconnect WS feed.
    """

    cache: ExchangeCache
    market_data: MarketDataProviderPort
    account_state: AccountStatePort
    order_stream: OrderStreamAdapter
    pairs: tuple[str, ...]
    account: str

    _status: FeedStatus = FeedStatus.DISCONNECTED
    _last_cursor: StreamCursor | None = None
    _ws_task: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def status(self) -> FeedStatus:
        return self._status

    async def bootstrap(self) -> None:
        """REST-fetch initial market data + account state into the cache.

        Must complete before the first tick is allowed to execute.
        """
        self._status = FeedStatus.BOOTSTRAPPING
        log.info("bootstrapping exchange cache via REST")
        await self.refresh_via_rest()
        log.info("bootstrap complete, cache populated")

    async def start(self) -> None:
        """Attempt to connect the WS feed; fall back to REST on failure."""
        try:
            await self.order_stream.connect()
            if self.order_stream.is_connected:
                self._status = FeedStatus.WS_ACTIVE
                self._last_cursor = await self.order_stream.get_cursor()
                log.info("WS feed connected")
                return
        except Exception:
            log.warning("WS connect failed, falling back to REST", exc_info=True)

        self._status = FeedStatus.REST_FALLBACK
        log.info("operating in REST fallback mode")

    async def refresh_via_rest(self) -> None:
        """Refresh cache from REST (used for bootstrap and fallback)."""
        prices = await self.market_data.fetch_latest_prices(self.pairs)
        account_snapshot = await self.account_state.get_account_snapshot(self.account)
        await self.cache.update_batch(prices, account_snapshot)

    async def check_and_recover_gaps(self) -> tuple[OrderEvent, ...]:
        """Detect and recover any gaps in the WS order stream.

        Returns a tuple of all recovered events.
        """
        if self._last_cursor is None:
            return ()

        gaps = await self.order_stream.detect_gaps(self._last_cursor)
        if not gaps:
            return ()

        all_recovered: list[OrderEvent] = []
        for gap in gaps:
            log.warning(
                "stream gap detected: %d -> %d",
                gap.start_sequence,
                gap.end_sequence,
            )
            recovered = await self.order_stream.recover_gap(gap)
            all_recovered.extend(recovered)
            log.info("recovered %d events for gap", len(recovered))

        self._last_cursor = await self.order_stream.get_cursor()

        if self._status == FeedStatus.WS_ACTIVE:
            await self.refresh_via_rest()

        return tuple(all_recovered)

    async def ensure_fresh(self) -> None:
        """Ensure the cache has current data, refreshing via REST if needed.

        Called before each tick to guarantee the context provider has data.
        Automatically activates REST fallback when WS is unavailable.
        """
        if self._status == FeedStatus.WS_ACTIVE and self.order_stream.is_connected:
            return

        if self._status == FeedStatus.WS_ACTIVE:
            log.warning("WS disconnected, switching to REST fallback")
            self._status = FeedStatus.REST_FALLBACK

        await self.refresh_via_rest()

    async def stop(self) -> None:
        """Disconnect the WS feed and clean up."""
        if self._ws_task is not None and not self._ws_task.done():
            self._ws_task.cancel()
            self._ws_task = None

        if self.order_stream.is_connected:
            await self.order_stream.disconnect()

        self._status = FeedStatus.DISCONNECTED
        log.info("exchange data feed stopped")

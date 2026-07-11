"""Exchange data feed: REST market/account snapshots + WS order-event status.

The WS user-data stream carries order events only — market prices and the
account snapshot always come from REST, refreshed before every tick. The
feed's status tracks whether the order-event socket is up; missed order
events are replayed by the consumer's recovery sweep, not here.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from dojiwick.domain.contracts.gateways.account_state import AccountStatePort
from dojiwick.domain.contracts.gateways.market_data_provider import MarketDataProviderPort
from dojiwick.infrastructure.exchange.cache import ExchangeCache

log = logging.getLogger(__name__)


class FeedStatus(StrEnum):
    """Current state of the data feed."""

    DISCONNECTED = "disconnected"
    BOOTSTRAPPING = "bootstrapping"
    WS_ACTIVE = "ws_active"
    REST_FALLBACK = "rest_fallback"


class OrderStreamAdapter(Protocol):
    """Connection lifecycle subset of OrderEventStreamPort used by the feed."""

    @property
    def stream_name(self) -> str: ...

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...


@dataclass(slots=True)
class ExchangeDataFeed:
    """REST-refreshed cache feed with WS order-stream lifecycle tracking.

    Lifecycle:
    1. ``bootstrap()`` — REST-fetch initial state into cache.
    2. ``start()`` — Attempt WS connect; on failure, note REST-fallback status.
    3. ``ensure_fresh()`` — REST refresh before every tick.
    4. ``stop()`` — Disconnect WS feed.
    """

    cache: ExchangeCache
    market_data: MarketDataProviderPort
    account_state: AccountStatePort
    order_stream: OrderStreamAdapter
    pairs: tuple[str, ...]
    account: str

    _status: FeedStatus = FeedStatus.DISCONNECTED

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
                log.info("WS feed connected")
                return
        except Exception:
            log.warning("WS connect failed, falling back to REST", exc_info=True)

        self._status = FeedStatus.REST_FALLBACK
        log.info("operating in REST fallback mode")

    async def refresh_via_rest(self) -> None:
        """Refresh cache from REST — the only source of prices and account state."""
        prices = await self.market_data.fetch_latest_prices(self.pairs)
        account_snapshot = await self.account_state.get_account_snapshot(self.account)
        await self.cache.update_batch(prices, account_snapshot)

    async def ensure_fresh(self) -> None:
        """Refresh the cache before a tick and track WS status transitions."""
        if self._status == FeedStatus.WS_ACTIVE and not self.order_stream.is_connected:
            log.warning("WS disconnected, switching to REST fallback")
            self._status = FeedStatus.REST_FALLBACK
        elif self._status == FeedStatus.REST_FALLBACK and self.order_stream.is_connected:
            log.info("WS reconnected, resuming WS-active status")
            self._status = FeedStatus.WS_ACTIVE

        await self.refresh_via_rest()

    async def stop(self) -> None:
        """Disconnect the WS feed and clean up."""
        if self.order_stream.is_connected:
            await self.order_stream.disconnect()

        self._status = FeedStatus.DISCONNECTED
        log.info("exchange data feed stopped")

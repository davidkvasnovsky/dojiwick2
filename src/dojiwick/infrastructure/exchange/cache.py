"""Thread-safe exchange data cache with consistent point-in-time snapshots.

The cache stores market data (prices) and account state (snapshots) populated
by WebSocket feeds or REST calls. Tick-loop consumers read atomic snapshots
via ``snapshot()`` which holds a read-lock to prevent torn reads during
concurrent writes.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.models.value_objects.account_state import AccountSnapshot
from dojiwick.domain.numerics import Price, to_price

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True, kw_only=True)
class CacheSnapshot:
    """Immutable point-in-time copy of the exchange cache."""

    prices: dict[str, Price]
    account: AccountSnapshot | None
    captured_at: datetime


@dataclass(slots=True)
class ExchangeCache:
    """Thread-safe exchange data cache with snapshot reads.

    Write path (WS feed / REST adapter) calls ``update_prices`` or
    ``update_account`` which hold an exclusive lock.  Read path (context
    provider) calls ``snapshot`` which takes the same lock to guarantee
    a consistent, non-torn view.
    """

    clock: ClockPort
    _prices: dict[str, Price] = field(default_factory=dict)
    _account: AccountSnapshot | None = None
    _last_update: datetime | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update_prices(self, prices: Mapping[str, Price | float]) -> None:
        """Replace the cached price map (called by WS feed or REST adapter)."""
        async with self._lock:
            normalized = {symbol: to_price(price) for symbol, price in prices.items()}
            self._prices.update(normalized)
            self._last_update = self.clock.now_utc()

    async def update_account(self, account: AccountSnapshot) -> None:
        """Replace the cached account snapshot."""
        async with self._lock:
            self._account = account
            self._last_update = self.clock.now_utc()

    async def update_batch(self, prices: Mapping[str, Price | float], account: AccountSnapshot) -> None:
        """Atomically update both prices and account under a single lock."""
        async with self._lock:
            normalized = {symbol: to_price(price) for symbol, price in prices.items()}
            self._prices.update(normalized)
            self._account = account
            self._last_update = self.clock.now_utc()

    async def snapshot(self) -> CacheSnapshot:
        """Return a consistent, immutable copy of the current cache state."""
        async with self._lock:
            return CacheSnapshot(
                prices=dict(self._prices),
                account=self._account,
                captured_at=self._last_update or self.clock.now_utc(),
            )

    async def clear(self) -> None:
        """Reset cache to empty state."""
        async with self._lock:
            self._prices.clear()
            self._account = None
            self._last_update = None

    @property
    def has_data(self) -> bool:
        """True if the cache has been populated at least once."""
        return self._last_update is not None

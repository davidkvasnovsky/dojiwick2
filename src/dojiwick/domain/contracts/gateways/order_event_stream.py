"""Order event stream gateway protocol with cursor-based replay."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_event import OrderEvent


@dataclass(slots=True, frozen=True, kw_only=True)
class StreamCursor:
    """Opaque cursor for resuming an event stream from a known position."""

    stream_name: str
    sequence: int
    timestamp_ms: int = 0


class OrderEventStreamPort(Protocol):
    """Real-time order event stream from the exchange with cursor replay."""

    @property
    def stream_name(self) -> str:
        """Adapter-provided name identifying this stream for cursor persistence."""
        ...

    async def connect(self) -> None:
        """Open the event stream connection."""
        ...

    async def disconnect(self) -> None:
        """Close the event stream connection."""
        ...

    def events(self) -> AsyncIterator[OrderEvent]:
        """Yield order events as they arrive."""
        ...

    def raw_updates(self) -> AsyncIterator[ExchangeOrderUpdate]:
        """Yield raw exchange order updates with full lifecycle data."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if the stream is currently connected."""
        ...

    def replay_from(self, cursor: StreamCursor) -> AsyncIterator[OrderEvent]:
        """Resume event delivery from the given cursor position."""
        ...

    async def get_cursor(self) -> StreamCursor:
        """Return the current stream position as a cursor."""
        ...

"""Order event stream test doubles."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from dojiwick.domain.contracts.gateways.order_event_stream import OrderEventStreamPort, StreamCursor
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_event import OrderEvent


@dataclass(slots=True)
class InMemoryOrderEventStream(OrderEventStreamPort):
    """In-memory order event stream for tests."""

    _events: list[OrderEvent] = field(default_factory=list)
    _replay_updates: list[ExchangeOrderUpdate] = field(default_factory=list)
    _raw_updates: list[ExchangeOrderUpdate] = field(default_factory=list)
    _connected: bool = False
    _sequence: int = 0
    _stream_name: str = "in_memory"
    _error_after: int = 0
    _error: Exception | None = None

    @property
    def stream_name(self) -> str:
        return self._stream_name

    async def connect(self) -> None:
        """Open the event stream connection."""
        self._connected = True

    async def disconnect(self) -> None:
        """Close the event stream connection."""
        self._connected = False

    async def raw_updates(self) -> AsyncIterator[ExchangeOrderUpdate]:
        """Yield raw exchange order updates."""
        for yielded, update in enumerate(self._raw_updates, start=1):
            self._sequence += 1
            yield update
            if self._error is not None and yielded >= self._error_after:
                raise self._error

    @property
    def is_connected(self) -> bool:
        """Return True if the stream is currently connected."""
        return self._connected

    async def replay_trades(self, symbol: str, start_time_ms: int) -> tuple[ExchangeOrderUpdate, ...]:
        """Return stored replay updates for the symbol."""
        return tuple(u for u in self._replay_updates if u.symbol == symbol)

    async def get_cursor(self) -> StreamCursor:
        """Return the current stream position as a cursor."""
        return StreamCursor(stream_name=self._stream_name, sequence=self._sequence)

    def push_event(self, event: OrderEvent) -> None:
        """Test helper: add an event to the stream."""
        self._events.append(event)

    def push_raw_update(self, update: ExchangeOrderUpdate) -> None:
        """Test helper: add a raw update to the stream."""
        self._raw_updates.append(update)

    def set_error_after(self, n: int, error: Exception) -> None:
        """Test helper: raise error after yielding n raw updates."""
        self._error_after = n
        self._error = error

    def push_replay_update(self, update: ExchangeOrderUpdate) -> None:
        """Test helper: add an update returned by replay_trades."""
        self._replay_updates.append(update)

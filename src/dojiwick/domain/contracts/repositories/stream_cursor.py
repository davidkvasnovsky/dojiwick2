"""Stream cursor repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord


class StreamCursorRepositoryPort(Protocol):
    """Stream cursor position tracking persistence."""

    async def get_cursor(self, stream_name: str) -> StreamCursorRecord | None:
        """Return the cursor for a stream, or None if absent."""
        ...

    async def set_cursor(self, cursor: StreamCursorRecord) -> None:
        """Insert or update the cursor for a stream."""
        ...

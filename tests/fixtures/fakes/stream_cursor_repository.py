"""Fake stream cursor repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord


@dataclass(slots=True)
class FakeStreamCursorRepo:
    """In-memory stream cursor repository."""

    cursors: dict[str, StreamCursorRecord] = field(default_factory=dict)

    async def get_cursor(self, stream_name: str) -> StreamCursorRecord | None:
        return self.cursors.get(stream_name)

    async def set_cursor(self, cursor: StreamCursorRecord) -> None:
        self.cursors[cursor.stream_name] = cursor

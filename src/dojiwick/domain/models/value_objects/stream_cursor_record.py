"""Stream cursor persistence record for the stream_cursors table."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class StreamCursorRecord:
    """Tracks the last-processed event position for a named stream."""

    stream_name: str
    last_event_id: str = ""
    last_event_time: datetime | None = None

    def __post_init__(self) -> None:
        if not self.stream_name:
            raise ValueError("stream_name must not be empty")

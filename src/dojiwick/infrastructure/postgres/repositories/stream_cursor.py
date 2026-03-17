"""PostgreSQL stream cursor repository."""

from dataclasses import dataclass
from datetime import UTC, datetime

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord

from dojiwick.infrastructure.postgres.connection import DbConnection

_UPSERT_SQL = """
INSERT INTO stream_cursors (stream_name, last_event_id, last_event_time, updated_at)
VALUES (%s, %s, %s, now())
ON CONFLICT (stream_name)
DO UPDATE SET
    last_event_id = EXCLUDED.last_event_id,
    last_event_time = EXCLUDED.last_event_time,
    updated_at = now()
"""

_SELECT_SQL = """
SELECT stream_name, last_event_id, last_event_time
FROM stream_cursors
WHERE stream_name = %s
"""


@dataclass(slots=True)
class PgStreamCursorRepository:
    """Persists stream cursor positions into PostgreSQL."""

    connection: DbConnection

    async def get_cursor(self, stream_name: str) -> StreamCursorRecord | None:
        """Return the cursor for a stream, or None if absent."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (stream_name,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get stream cursor: {exc}") from exc
        if row is None:
            return None
        (name, last_event_id, last_event_time) = row
        if isinstance(last_event_time, str):
            last_event_time = datetime.fromisoformat(last_event_time)
        if isinstance(last_event_time, datetime) and last_event_time.tzinfo is None:
            last_event_time = last_event_time.replace(tzinfo=UTC)
        return StreamCursorRecord(
            stream_name=str(name),
            last_event_id=str(last_event_id),
            last_event_time=last_event_time if isinstance(last_event_time, datetime) else None,
        )

    async def set_cursor(self, cursor: StreamCursorRecord) -> None:
        """Insert or update the cursor for a stream."""
        row = (
            cursor.stream_name,
            cursor.last_event_id,
            cursor.last_event_time.isoformat() if cursor.last_event_time else None,
        )
        try:
            async with self.connection.cursor() as db_cursor:
                await db_cursor.execute(_UPSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to set stream cursor: {exc}") from exc

"""PostgreSQL tick repository."""

from dataclasses import dataclass

from dojiwick.domain.enums import TickStatus
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.tick_record import TickRecord
from dojiwick.infrastructure.postgres.connection import DbConnection

_TRY_INSERT_SQL = """
INSERT INTO ticks (
    tick_id, tick_time, config_hash, schema_ver,
    inputs_hash, intent_hash, ops_hash,
    authority, status, batch_size, duration_ms, error_message
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (tick_id) DO UPDATE
SET inputs_hash = EXCLUDED.inputs_hash,
    batch_size = EXCLUDED.batch_size,
    status = EXCLUDED.status
WHERE ticks.status != 'completed'
RETURNING tick_id
"""

_RECOVER_STALE_SQL = """
UPDATE ticks
SET status = 'failed', error_message = 'recovered_stale_started', updated_at = now()
WHERE status = 'started' AND created_at < now() - interval '1 second' * %s
"""

_UPDATE_STATUS_SQL = """
UPDATE ticks
SET status = %s, intent_hash = %s, ops_hash = %s,
    duration_ms = %s, error_message = %s, updated_at = now()
WHERE tick_id = %s
"""


@dataclass(slots=True)
class PgTickRepository:
    """Persists tick lifecycle records into PostgreSQL."""

    connection: DbConnection

    async def try_insert(self, record: TickRecord) -> bool:
        """Atomically insert or detect completed-tick dedup via ON CONFLICT.

        Returns True if the row was inserted/updated (not completed).
        Returns False if a completed tick already exists (dedup).
        """
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    _TRY_INSERT_SQL,
                    (
                        record.tick_id,
                        record.tick_time.isoformat(),
                        record.config_hash,
                        record.schema_ver,
                        record.inputs_hash,
                        record.intent_hash,
                        record.ops_hash,
                        record.authority.value,
                        record.status.value,
                        record.batch_size,
                        record.duration_ms,
                        record.error_message,
                    ),
                )
                row = await cursor.fetchone()
            await self.connection.commit()
            return row is not None
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert tick record: {exc}") from exc

    async def recover_stale_started(self, stale_threshold_sec: int = 300) -> int:
        """Mark stale STARTED ticks as FAILED. Returns count of recovered ticks."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_RECOVER_STALE_SQL, (stale_threshold_sec,))
                count = cursor.rowcount
            await self.connection.commit()
            return count
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to recover stale ticks: {exc}") from exc

    async def update_status(
        self,
        tick_id: str,
        status: TickStatus,
        *,
        intent_hash: str = "",
        ops_hash: str = "",
        duration_ms: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update tick status and optional hash fields."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    _UPDATE_STATUS_SQL,
                    (status.value, intent_hash, ops_hash, duration_ms, error_message, tick_id),
                )
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to update tick status: {exc}") from exc

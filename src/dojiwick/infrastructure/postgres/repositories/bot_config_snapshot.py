"""PostgreSQL bot config snapshot repository."""

from dataclasses import dataclass

from dojiwick.domain.errors import AdapterError

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO bot_config_snapshots (config_hash, config_json, snapshot_at)
VALUES (%s, %s::jsonb, now())
"""

_SELECT_LATEST_SQL = """
SELECT config_hash, config_json::text FROM bot_config_snapshots
ORDER BY snapshot_at DESC LIMIT 1
"""


@dataclass(slots=True)
class PgBotConfigSnapshotRepository:
    """Persists configuration snapshots into PostgreSQL."""

    connection: DbConnection

    async def record_snapshot(self, config_hash: str, config_json: str) -> None:
        """Persist a configuration snapshot."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, (config_hash, config_json))
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record config snapshot: {exc}") from exc

    async def get_latest(self) -> tuple[str, str] | None:
        """Return the latest snapshot as (hash, json), or None if absent."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_LATEST_SQL)
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get config snapshot: {exc}") from exc
        if row is None:
            return None
        return (str(row[0]), str(row[1]))

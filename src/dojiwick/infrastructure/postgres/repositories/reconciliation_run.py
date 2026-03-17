"""PostgreSQL reconciliation run repository."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.reconciliation_run import ReconciliationRun

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO reconciliation_runs (
    run_type, status, orphaned_db_count, orphaned_exchange_count,
    mismatch_count, resolved_count, divergences, started_at, completed_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_LATEST_SQL = """
SELECT id, run_type, status, orphaned_db_count, orphaned_exchange_count,
       mismatch_count, resolved_count, divergences, started_at, completed_at
FROM reconciliation_runs
{where_clause}
ORDER BY created_at DESC
LIMIT 1
"""


def _row_to_run(row: tuple[object, ...]) -> ReconciliationRun:
    """Map a DB row to ReconciliationRun."""
    (
        db_id,
        run_type,
        status,
        orphaned_db_count,
        orphaned_exchange_count,
        mismatch_count,
        resolved_count,
        divergences,
        started_at,
        completed_at,
    ) = row
    for ts_name, ts_val in [("started_at", started_at), ("completed_at", completed_at)]:
        if isinstance(ts_val, str):
            ts_val = datetime.fromisoformat(ts_val)
        if isinstance(ts_val, datetime) and ts_val.tzinfo is None:
            ts_val = ts_val.replace(tzinfo=UTC)
        if ts_name == "started_at":
            started_at = ts_val
        else:
            completed_at = ts_val
    div_dict: dict[str, object] | None = None
    if divergences is not None:
        if isinstance(divergences, dict):
            div_dict = cast(dict[str, object], divergences)
        elif isinstance(divergences, str):
            div_dict = cast(dict[str, object], json.loads(divergences))
    return ReconciliationRun(
        id=int(str(db_id)),
        run_type=str(run_type),
        status=str(status),
        orphaned_db_count=int(str(orphaned_db_count)),
        orphaned_exchange_count=int(str(orphaned_exchange_count)),
        mismatch_count=int(str(mismatch_count)),
        resolved_count=int(str(resolved_count)),
        divergences=div_dict,
        started_at=started_at if isinstance(started_at, datetime) else None,
        completed_at=completed_at if isinstance(completed_at, datetime) else None,
    )


@dataclass(slots=True)
class PgReconciliationRunRepository:
    """Persists reconciliation run results into PostgreSQL."""

    connection: DbConnection

    async def insert_run(self, run: ReconciliationRun) -> int:
        """Persist a reconciliation run and return the DB-assigned id."""
        div_json = json.dumps(run.divergences) if run.divergences else None
        row = (
            run.run_type,
            run.status,
            run.orphaned_db_count,
            run.orphaned_exchange_count,
            run.mismatch_count,
            run.resolved_count,
            div_json,
            run.started_at.isoformat() if run.started_at else None,
            run.completed_at.isoformat() if run.completed_at else None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert reconciliation run: {exc}") from exc
        if result is None:
            raise AdapterError("INSERT reconciliation_runs returned no id")
        return int(result[0])

    async def get_latest(self, run_type: str | None = None) -> ReconciliationRun | None:
        """Return the most recent reconciliation run, optionally filtered by type."""
        if run_type is not None:
            where_clause = "WHERE run_type = %s"
            params: tuple[object, ...] = (run_type,)
        else:
            where_clause = ""
            params = ()
        sql = _SELECT_LATEST_SQL.format(where_clause=where_clause)
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(sql, params if params else None)
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get latest reconciliation run: {exc}") from exc
        if row is None:
            return None
        return _row_to_run(row)

"""PostgreSQL system event repository."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from dojiwick.domain.enums import AuditSeverity
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.system_event import SystemEvent

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO system_event_log (component, severity, message, correlation_id, context, occurred_at)
VALUES (%s, %s::system_event_severity, %s, %s, %s, %s)
"""

_SELECT_SQL = """
SELECT component, severity, message, correlation_id, context, occurred_at
FROM system_event_log
{where_clause}
ORDER BY occurred_at DESC
"""


def _row_to_event(row: tuple[object, ...]) -> SystemEvent:
    """Map a DB row to SystemEvent."""
    (component, severity, message, correlation_id, context, occurred_at) = row
    if isinstance(occurred_at, str):
        occurred_at = datetime.fromisoformat(occurred_at)
    if isinstance(occurred_at, datetime) and occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    ctx: dict[str, object] | None = None
    if context is not None:
        if isinstance(context, dict):
            ctx = cast(dict[str, object], context)
        elif isinstance(context, str):
            ctx = cast(dict[str, object], json.loads(context))
    return SystemEvent(
        component=str(component),
        severity=AuditSeverity(str(severity)),
        message=str(message),
        correlation_id=str(correlation_id),
        context=ctx,
        occurred_at=occurred_at if isinstance(occurred_at, datetime) else None,
    )


@dataclass(slots=True)
class PgSystemEventRepository:
    """Persists system events into PostgreSQL."""

    connection: DbConnection

    async def record_event(self, event: SystemEvent) -> None:
        """Persist a system event."""
        ctx_json = json.dumps(event.context) if event.context else None
        occurred = event.occurred_at.isoformat() if event.occurred_at else None
        row = (
            event.component,
            event.severity.value,
            event.message,
            event.correlation_id,
            ctx_json,
            occurred,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record system event: {exc}") from exc

    async def get_events(self, component: str | None = None, since: datetime | None = None) -> tuple[SystemEvent, ...]:
        """Return system events, optionally filtered by component and/or time."""
        conditions: list[str] = []
        params: list[object] = []
        if component is not None:
            conditions.append("component = %s")
            params.append(component)
        if since is not None:
            conditions.append("occurred_at >= %s")
            params.append(since.isoformat())
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = _SELECT_SQL.format(where_clause=where_clause)
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(sql, tuple(params) if params else None)
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get system events: {exc}") from exc
        return tuple(_row_to_event(r) for r in rows)

"""PostgreSQL audit log."""

import json
from dataclasses import dataclass

from dojiwick.domain.enums import AuditSeverity
from dojiwick.domain.errors import AdapterError

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO audit_log (severity, event_type, message, context)
VALUES (%s::audit_severity, %s, %s, %s::jsonb)
"""


@dataclass(slots=True)
class PgAuditLog:
    """Persists audit events into PostgreSQL."""

    connection: DbConnection

    async def log_event(
        self,
        severity: AuditSeverity,
        event_type: str,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    _INSERT_SQL,
                    (severity.value, event_type, message, json.dumps(context) if context else None),
                )
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to log audit event: {exc}") from exc

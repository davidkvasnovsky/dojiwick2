"""Audit log protocol."""

from typing import Protocol

from dojiwick.domain.enums import AuditSeverity


class AuditLogPort(Protocol):
    """Immutable event log for operational audit trail."""

    async def log_event(
        self,
        severity: AuditSeverity,
        event_type: str,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        """Persist an audit event."""
        ...

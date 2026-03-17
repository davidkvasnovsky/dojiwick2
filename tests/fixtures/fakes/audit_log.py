"""Audit log test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.enums import AuditSeverity


@dataclass(slots=True)
class CapturingAuditLog:
    """Captures all audit events for assertion."""

    events: list[dict[str, object]] = field(default_factory=list)

    async def log_event(
        self,
        severity: AuditSeverity,
        event_type: str,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "severity": severity,
                "event_type": event_type,
                "message": message,
                "context": context,
            }
        )


class NullAuditLog:
    """Discards all audit events."""

    async def log_event(
        self,
        severity: AuditSeverity,
        event_type: str,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        del severity, event_type, message, context

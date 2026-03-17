"""Notification test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.enums import AuditSeverity


@dataclass(slots=True)
class CapturingNotification:
    """Captures all alerts for assertion."""

    alerts: list[tuple[AuditSeverity, str, dict[str, object] | None]] = field(default_factory=list)

    async def send_alert(
        self, severity: AuditSeverity, message: str, *, context: dict[str, object] | None = None
    ) -> None:
        self.alerts.append((severity, message, context))

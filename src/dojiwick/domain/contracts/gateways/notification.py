"""Notification port for alerting on critical events."""

from typing import Protocol

from dojiwick.domain.enums import AuditSeverity


class NotificationPort(Protocol):
    """Sends operational alerts (double-fills, consecutive errors, liquidation risk)."""

    async def send_alert(
        self, severity: AuditSeverity, message: str, *, context: dict[str, object] | None = None
    ) -> None:
        """Dispatch an alert to the configured notification channel."""
        ...

"""Log-based notification adapter."""

import logging

from dojiwick.domain.enums import AuditSeverity

log = logging.getLogger(__name__)

_SEVERITY_MAP: dict[AuditSeverity, int] = {
    AuditSeverity.INFO: logging.INFO,
    AuditSeverity.WARNING: logging.WARNING,
    AuditSeverity.CRITICAL: logging.CRITICAL,
}


class LogNotification:
    """Implements NotificationPort by mapping alerts to log levels."""

    __slots__ = ()

    async def send_alert(
        self, severity: AuditSeverity, message: str, *, context: dict[str, object] | None = None
    ) -> None:
        level = _SEVERITY_MAP.get(severity, logging.WARNING)
        if context:
            log.log(level, "%s context=%s", message, context)
        else:
            log.log(level, "%s", message)

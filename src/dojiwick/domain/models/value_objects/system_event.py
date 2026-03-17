"""System-level event value object for structured audit logging."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.enums import AuditSeverity


@dataclass(slots=True, frozen=True, kw_only=True)
class SystemEvent:
    """Immutable system event for audit and observability."""

    component: str
    severity: AuditSeverity
    message: str
    correlation_id: str = ""
    context: dict[str, object] | None = None
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("component must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")
        if self.occurred_at is not None and self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware if set")

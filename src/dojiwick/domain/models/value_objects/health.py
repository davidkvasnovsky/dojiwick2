"""Health status domain model."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class HealthStatus:
    """Structured health check result."""

    healthy: bool
    db_connected: bool
    last_tick_at: datetime | None
    consecutive_errors: int
    details: dict[str, str]

    def __post_init__(self) -> None:
        if self.consecutive_errors < 0:
            raise ValueError("consecutive_errors must be non-negative")
        if self.last_tick_at is not None and self.last_tick_at.tzinfo is None:
            raise ValueError("last_tick_at must be timezone-aware")

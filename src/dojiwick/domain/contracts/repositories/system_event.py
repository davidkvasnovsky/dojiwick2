"""System event repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.system_event import SystemEvent


class SystemEventRepositoryPort(Protocol):
    """System event persistence and retrieval."""

    async def record_event(self, event: SystemEvent) -> None:
        """Persist a system event."""
        ...

    async def get_events(self, component: str | None = None, since: datetime | None = None) -> tuple[SystemEvent, ...]:
        """Return system events, optionally filtered by component and/or time."""
        ...

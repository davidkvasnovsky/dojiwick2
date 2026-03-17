"""Fake system event repository for tests."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.system_event import SystemEvent


@dataclass(slots=True)
class FakeSystemEventRepository:
    """In-memory system event repository for test assertions."""

    events: list[SystemEvent] = field(default_factory=list)

    async def record_event(self, event: SystemEvent) -> None:
        self.events.append(event)

    async def get_events(self, component: str | None = None, since: datetime | None = None) -> tuple[SystemEvent, ...]:
        result = self.events
        if component is not None:
            result = [e for e in result if e.component == component]
        if since is not None:
            result = [e for e in result if e.occurred_at is not None and e.occurred_at >= since]
        return tuple(result)

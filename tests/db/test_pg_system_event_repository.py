"""Integration tests for PgSystemEventRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dojiwick.domain.enums import AuditSeverity
from dojiwick.domain.models.value_objects.system_event import SystemEvent

pytestmark = pytest.mark.db


def _make_event(component: str) -> SystemEvent:
    return SystemEvent(
        component=component,
        severity=AuditSeverity.INFO,
        message=f"test event from {component}",
        occurred_at=datetime.now(UTC),
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.system_event import PgSystemEventRepository

    return PgSystemEventRepository(connection=db_connection)


async def test_record_and_get(repo: Any, clean_tables: None) -> None:
    event = _make_event("risk_engine")
    await repo.record_event(event)

    events = await repo.get_events()
    assert len(events) == 1
    assert events[0].component == "risk_engine"


async def test_filter_by_component(repo: Any, clean_tables: None) -> None:
    await repo.record_event(_make_event("tick_loop"))
    await repo.record_event(_make_event("risk_engine"))

    events = await repo.get_events(component="tick_loop")
    assert len(events) == 1
    assert events[0].component == "tick_loop"


async def test_get_empty(repo: Any, clean_tables: None) -> None:
    events = await repo.get_events()
    assert events == ()

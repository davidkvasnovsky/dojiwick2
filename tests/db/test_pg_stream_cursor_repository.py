"""Integration tests for PgStreamCursorRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.stream_cursor import PgStreamCursorRepository

    return PgStreamCursorRepository(connection=db_connection)


async def test_set_and_get_cursor(repo: Any, clean_tables: None) -> None:
    cursor = StreamCursorRecord(
        stream_name="order-events",
        last_event_id="evt-42",
        last_event_time=datetime.now(UTC),
    )
    await repo.set_cursor(cursor)

    loaded = await repo.get_cursor("order-events")
    assert loaded is not None
    assert loaded.stream_name == "order-events"
    assert loaded.last_event_id == "evt-42"


async def test_upsert_updates_cursor(repo: Any, clean_tables: None) -> None:
    cursor1 = StreamCursorRecord(
        stream_name="trades",
        last_event_id="evt-1",
        last_event_time=datetime.now(UTC),
    )
    await repo.set_cursor(cursor1)

    cursor2 = StreamCursorRecord(
        stream_name="trades",
        last_event_id="evt-99",
        last_event_time=datetime.now(UTC),
    )
    await repo.set_cursor(cursor2)

    loaded = await repo.get_cursor("trades")
    assert loaded is not None
    assert loaded.last_event_id == "evt-99"


async def test_get_cursor_not_found(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_cursor("nonexistent-stream")
    assert loaded is None

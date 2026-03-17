"""Integration tests for PgOrderEventRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.enums import OrderEventType
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def test_order_request_id(db_cursor: DbCursor, test_instrument_id: int) -> int:
    """Insert parent rows (instrument → order_request) and return the order_request id."""
    await db_cursor.execute(
        """
        INSERT INTO order_requests (
            client_order_id, instrument_id, account, side, order_type, quantity
        ) VALUES ('cli-1', %s, 'test-acct', 'buy', 'market', 1.0)
        RETURNING id
        """,
        (test_instrument_id,),
    )
    row = await db_cursor.fetchone()
    assert row is not None
    return int(str(row[0]))


def _make_event(order_id: int) -> OrderEvent:
    return OrderEvent(
        order_id=order_id,
        event_type=OrderEventType.PLACED,
        occurred_at=datetime.now(UTC),
        exchange_order_id="exch-1",
        filled_quantity=Decimal(0),
        fees_usd=Decimal(0),
        detail="test event",
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.order_event import PgOrderEventRepository

    return PgOrderEventRepository(connection=db_connection)


async def test_record_and_get_for_order(repo: Any, test_order_request_id: int) -> None:
    event = _make_event(test_order_request_id)
    await repo.record_event(event)

    events = await repo.get_events_for_order(test_order_request_id)
    assert len(events) == 1
    assert events[0].order_id == test_order_request_id
    assert events[0].event_type == OrderEventType.PLACED


async def test_get_for_order_empty(repo: Any, test_order_request_id: int) -> None:
    events = await repo.get_events_for_order(test_order_request_id)
    assert events == ()


async def test_get_events_since(repo: Any, test_order_request_id: int) -> None:
    before = datetime.now(UTC)
    event = _make_event(test_order_request_id)
    await repo.record_event(event)

    events = await repo.get_events_since(before)
    assert len(events) >= 1

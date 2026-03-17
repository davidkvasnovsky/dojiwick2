"""Integration tests for PgPositionEventRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import PositionEventType, PositionSide
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord, PositionLeg

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.position_event import PgPositionEventRepository

    return PgPositionEventRepository(connection=db_connection)


@pytest.fixture
def leg_repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.position_leg import PgPositionLegRepository

    return PgPositionLegRepository(connection=db_connection)


async def test_record_and_get_events(repo: Any, leg_repo: Any, test_instrument_id: int) -> None:
    leg = PositionLeg(
        account="test-account",
        instrument_id=test_instrument_id,
        position_side=PositionSide.LONG,
        quantity=Decimal("1"),
        entry_price=Decimal("50000"),
        opened_at=datetime.now(UTC),
    )
    leg_id = await leg_repo.insert_leg(leg)

    event = PositionEventRecord(
        position_leg_id=leg_id,
        event_type=PositionEventType.OPEN,
        quantity=Decimal("1"),
        price=Decimal("50000"),
        occurred_at=datetime.now(UTC),
    )
    event_id = await repo.record_event(event)
    assert event_id > 0

    events = await repo.get_events_for_leg(leg_id)
    assert len(events) == 1
    assert events[0].event_type is PositionEventType.OPEN
    assert events[0].quantity == Decimal("1")


async def test_get_events_empty(repo: Any, test_instrument_id: int) -> None:
    events = await repo.get_events_for_leg(999999)
    assert events == ()

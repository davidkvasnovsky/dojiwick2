"""Tests that verify transaction atomicity (commit/rollback behavior)."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.models.value_objects.position_leg import PositionLeg

pytestmark = pytest.mark.db


async def test_rollback_undoes_insert(db_connection: Any, db_cursor: Any, test_instrument_id: int) -> None:
    """A rolled-back insert should not be visible after rollback."""
    from dojiwick.infrastructure.postgres.repositories.position_leg import PgPositionLegRepository

    repo = PgPositionLegRepository(connection=db_connection)

    leg = PositionLeg(
        account="rollback-test",
        instrument_id=test_instrument_id,
        position_side=PositionSide.LONG,
        quantity=Decimal("1"),
        entry_price=Decimal("50000"),
        opened_at=datetime.now(UTC),
    )
    leg_id = await repo.insert_leg(leg)
    assert leg_id > 0

    # Rollback the transaction
    await db_connection.rollback()

    # The row should not be visible
    await db_cursor.execute("SELECT COUNT(*) FROM position_legs WHERE account = 'rollback-test'")
    row: tuple[object, ...] | None = await db_cursor.fetchone()
    assert row is not None
    assert int(str(row[0])) == 0


async def test_committed_data_persists(db_connection: Any, db_cursor: Any, test_instrument_id: int) -> None:
    """Committed data should survive and be queryable."""
    from dojiwick.infrastructure.postgres.repositories.stream_cursor import PgStreamCursorRepository
    from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord

    repo = PgStreamCursorRepository(connection=db_connection)

    cursor = StreamCursorRecord(
        stream_name="atomicity-test",
        last_event_id="evt-1",
        last_event_time=datetime.now(UTC),
    )
    await repo.set_cursor(cursor)
    # set_cursor already commits internally

    loaded = await repo.get_cursor("atomicity-test")
    assert loaded is not None
    assert loaded.last_event_id == "evt-1"

"""Integration tests for PgAdaptiveSelectionRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveSelectionEvent
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def test_leg_id(db_cursor: DbCursor, test_instrument_id: int) -> int:
    """Insert FK chain: instrument → position_leg → adaptive_config."""
    await db_cursor.execute(
        """
        INSERT INTO position_legs (account, instrument_id, position_side, quantity, entry_price)
        VALUES ('test-acct', %s, 'net', 1.0, 100.0)
        RETURNING id
        """,
        (test_instrument_id,),
    )
    row = await db_cursor.fetchone()
    assert row is not None
    leg_id = int(str(row[0]))

    await db_cursor.execute("""
        INSERT INTO adaptive_configs (config_idx, params_json)
        VALUES (0, '{"k": 1}')
        ON CONFLICT DO NOTHING
    """)

    return leg_id


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.adaptive_selection import PgAdaptiveSelectionRepository

    return PgAdaptiveSelectionRepository(connection=db_connection)


def _make_selection(leg_id: int) -> AdaptiveSelectionEvent:
    return AdaptiveSelectionEvent(
        position_leg_id=leg_id,
        arm=AdaptiveArmKey(regime_idx=1, config_idx=0),
        selected_at=datetime.now(UTC),
    )


async def test_record_and_get_selection(repo: Any, test_leg_id: int) -> None:
    event = _make_selection(test_leg_id)
    await repo.record_selection(event)

    loaded = await repo.get_selection(test_leg_id)
    assert loaded is not None
    assert loaded.position_leg_id == test_leg_id
    assert loaded.arm.regime_idx == 1
    assert loaded.arm.config_idx == 0


async def test_get_selection_not_found(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_selection(999999)
    assert loaded is None

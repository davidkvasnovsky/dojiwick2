"""Integration tests for PgAdaptiveOutcomeRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveOutcomeEvent
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def test_selection_id(db_cursor: DbCursor, test_instrument_id: int) -> int:
    """Insert FK chain: instrument → position_leg → adaptive_config → adaptive_selection."""
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

    await db_cursor.execute(
        """
        INSERT INTO adaptive_selections (position_leg_id, regime_idx, config_idx)
        VALUES (%s, 1, 0)
        """,
        (leg_id,),
    )
    return leg_id


def _make_outcome(leg_id: int) -> AdaptiveOutcomeEvent:
    return AdaptiveOutcomeEvent(
        position_leg_id=leg_id,
        arm=AdaptiveArmKey(regime_idx=1, config_idx=0),
        reward=0.75,
        observed_at=datetime.now(UTC),
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.adaptive_outcome import PgAdaptiveOutcomeRepository

    return PgAdaptiveOutcomeRepository(connection=db_connection)


async def test_record_and_get_outcome(repo: Any, test_selection_id: int) -> None:
    event = _make_outcome(test_selection_id)
    await repo.record_outcome(event)

    loaded = await repo.get_outcome(test_selection_id)
    assert loaded is not None
    assert loaded.position_leg_id == test_selection_id
    assert loaded.reward == 0.75


async def test_get_outcome_not_found(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_outcome(999999)
    assert loaded is None

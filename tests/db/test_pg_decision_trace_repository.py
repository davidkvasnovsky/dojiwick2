"""Integration tests for PgDecisionTraceRepository."""

from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def test_tick_id(db_cursor: DbCursor, clean_tables: None) -> str:
    """Insert a parent tick row for FK satisfaction."""
    await db_cursor.execute("""
        INSERT INTO ticks (tick_id, tick_time, config_hash, inputs_hash)
        VALUES ('tick-trace-1', now(), 'cfg', 'inp')
    """)
    return "tick-trace-1"


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.decision_trace import PgDecisionTraceRepository

    return PgDecisionTraceRepository(connection=db_connection)


async def test_insert_batch(repo: Any, test_tick_id: str) -> None:
    traces = (
        DecisionTrace(tick_id=test_tick_id, step_name="regime", step_seq=1, artifacts={"state": 1}),
        DecisionTrace(tick_id=test_tick_id, step_name="strategy", step_seq=2),
    )
    await repo.insert_batch(traces)


async def test_insert_empty_batch(repo: Any, clean_tables: None) -> None:
    await repo.insert_batch(())

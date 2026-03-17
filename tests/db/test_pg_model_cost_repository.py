"""Integration tests for PgModelCostRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.models.value_objects.model_cost import ModelCostRecord
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def test_tick_id(db_cursor: DbCursor, clean_tables: None) -> str:
    """Insert a tick row for the FK constraint."""
    tick_id = "test-tick-001"
    await db_cursor.execute(
        """
        INSERT INTO ticks (tick_id, tick_time, config_hash, inputs_hash)
        VALUES (%s, now(), 'hash', 'inputs')
        """,
        (tick_id,),
    )
    return tick_id


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.model_cost import PgModelCostRepository

    return PgModelCostRepository(connection=db_connection)


def _make_record(tick_id: str) -> ModelCostRecord:
    return ModelCostRecord(
        tick_id=tick_id,
        model="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=Decimal("0.00350000"),
        purpose="veto",
        created_at=datetime.now(UTC),
    )


async def test_record_cost(repo: Any, test_tick_id: str, db_cursor: DbCursor) -> None:
    record = _make_record(test_tick_id)
    await repo.record_cost(record)

    await db_cursor.execute("SELECT tick_id, model, input_tokens, output_tokens, cost_usd, purpose FROM model_costs")
    row = await db_cursor.fetchone()
    assert row is not None
    assert str(row[0]) == test_tick_id
    assert str(row[1]) == "claude-sonnet-4-20250514"
    assert int(str(row[2])) == 1000
    assert int(str(row[3])) == 200
    assert float(str(row[4])) == pytest.approx(0.0035)  # pyright: ignore[reportUnknownMemberType]
    assert str(row[5]) == "veto"


async def test_batch_record_costs(repo: Any, test_tick_id: str, db_cursor: DbCursor) -> None:
    records = tuple(
        ModelCostRecord(
            tick_id=test_tick_id,
            model=f"model-{i}",
            input_tokens=1000 * (i + 1),
            output_tokens=100 * (i + 1),
            cost_usd=Decimal(f"0.00{i + 1}00000"),
            purpose="veto",
            created_at=datetime.now(UTC),
        )
        for i in range(3)
    )
    await repo.batch_record_costs(records)

    await db_cursor.execute("SELECT model FROM model_costs ORDER BY model")
    rows = await db_cursor.fetchall()
    assert len(rows) == 3
    assert [str(r[0]) for r in rows] == ["model-0", "model-1", "model-2"]

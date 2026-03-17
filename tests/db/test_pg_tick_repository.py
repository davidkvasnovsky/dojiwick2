"""Integration tests for PgTickRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dojiwick.domain.enums import DecisionAuthority, TickStatus
from dojiwick.domain.models.value_objects.tick_record import TickRecord

pytestmark = pytest.mark.db


def _make_record(tick_id: str, *, status: TickStatus = TickStatus.STARTED) -> TickRecord:
    return TickRecord(
        tick_id=tick_id,
        tick_time=datetime.now(UTC),
        config_hash="cfg-abc",
        inputs_hash="inp-123",
        authority=DecisionAuthority.DETERMINISTIC_ONLY,
        status=status,
        batch_size=5,
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.tick import PgTickRepository

    return PgTickRepository(connection=db_connection)


async def test_try_insert_new_tick(repo: Any, clean_tables: None) -> None:
    record = _make_record("tick-001")
    result = await repo.try_insert(record)
    assert result is True


async def test_try_insert_dedup_completed(repo: Any, clean_tables: None) -> None:
    record = _make_record("tick-002")
    assert await repo.try_insert(record) is True

    await repo.update_status("tick-002", TickStatus.COMPLETED, intent_hash="ih", ops_hash="oh")

    dup = _make_record("tick-002")
    assert await repo.try_insert(dup) is False


async def test_update_status(repo: Any, clean_tables: None) -> None:
    record = _make_record("tick-003")
    await repo.try_insert(record)

    await repo.update_status(
        "tick-003",
        TickStatus.FAILED,
        error_message="boom",
        duration_ms=42,
    )

    # Verify via raw cursor that the update persisted.
    async with repo.connection.cursor() as cur:
        await cur.execute("SELECT status, error_message FROM ticks WHERE tick_id = %s", ("tick-003",))
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert row[1] == "boom"

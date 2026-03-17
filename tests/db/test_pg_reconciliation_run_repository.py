"""Integration tests for PgReconciliationRunRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.reconciliation_run import ReconciliationRun

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.reconciliation_run import PgReconciliationRunRepository

    return PgReconciliationRunRepository(connection=db_connection)


async def test_insert_and_get_latest(repo: Any, clean_tables: None) -> None:
    run = ReconciliationRun(
        run_type="startup",
        status="completed",
        orphaned_db_count=0,
        orphaned_exchange_count=0,
        mismatch_count=1,
        resolved_count=0,
        divergences={"BTC/USDC": "quantity_mismatch"},
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    db_id = await repo.insert_run(run)
    assert db_id > 0

    loaded = await repo.get_latest()
    assert loaded is not None
    assert loaded.run_type == "startup"
    assert loaded.mismatch_count == 1
    assert loaded.divergences is not None
    assert "BTC/USDC" in loaded.divergences


async def test_get_latest_filtered_by_type(repo: Any, clean_tables: None) -> None:
    startup = ReconciliationRun(
        run_type="startup",
        started_at=datetime.now(UTC),
    )
    periodic = ReconciliationRun(
        run_type="periodic",
        started_at=datetime.now(UTC),
    )
    await repo.insert_run(startup)
    await repo.insert_run(periodic)

    loaded = await repo.get_latest(run_type="periodic")
    assert loaded is not None
    assert loaded.run_type == "periodic"


async def test_get_latest_returns_none(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_latest()
    assert loaded is None

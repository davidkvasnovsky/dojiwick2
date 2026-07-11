"""Integration tests for PgPerformanceRepository."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.performance import PgPerformanceRepository

    return PgPerformanceRepository(connection=db_connection)


def _make_snapshot(observed_at: datetime | None = None) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        observed_at=observed_at or datetime.now(UTC),
        equity_usd=Decimal("10000"),
        unrealized_pnl_usd=Decimal("50"),
        realized_pnl_usd=Decimal("100"),
        open_positions=2,
        drawdown_pct=Decimal("1.5"),
    )


async def test_record_and_get_snapshots(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    await repo.record_snapshot(_make_snapshot(now))
    got = await repo.get_snapshots(now - timedelta(minutes=1), now + timedelta(minutes=1))
    assert len(got) == 1
    assert got[0].equity_usd == Decimal("10000")


async def test_get_snapshots_empty_range(repo: Any, clean_tables: None) -> None:
    now = datetime.now(UTC)
    got = await repo.get_snapshots(now - timedelta(days=2), now - timedelta(days=1))
    assert got == ()

"""Integration tests for PgPerformanceRepository."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.performance import PgPerformanceRepository

    return PgPerformanceRepository(connection=db_connection)


def _make_snapshot() -> PerformanceSnapshot:
    return PerformanceSnapshot(
        observed_at=datetime.now(UTC),
        equity_usd=Decimal("10000"),
        unrealized_pnl_usd=Decimal("50"),
        realized_pnl_usd=Decimal("100"),
        open_positions=2,
        drawdown_pct=Decimal("1.5"),
    )


async def test_insert_and_get_latest(repo: Any, clean_tables: None) -> None:
    snapshot = _make_snapshot()
    await repo.insert(snapshot)
    latest = await repo.get_latest()
    assert latest is not None
    assert latest.equity_usd == Decimal("10000")


async def test_get_latest_returns_none_when_empty(repo: Any, clean_tables: None) -> None:
    latest = await repo.get_latest()
    assert latest is None

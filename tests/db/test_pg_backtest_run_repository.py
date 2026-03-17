"""Integration tests for PgBacktestRunRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dojiwick.domain.models.value_objects.backtest_run import BacktestRunRecord
from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary

pytestmark = pytest.mark.db


def _make_record(config_hash: str) -> BacktestRunRecord:
    return BacktestRunRecord(
        config_hash=config_hash,
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 6, 1, tzinfo=UTC),
        interval="1h",
        pairs=("BTCUSDC",),
        target_ids=("BTCUSDC",),
        venue="binance",
        product="usd_c",
        summary=BacktestSummary(
            trades=100,
            total_pnl_usd=500.0,
            win_rate=0.55,
            expectancy_usd=5.0,
            sharpe_like=1.2,
            max_drawdown_pct=0.08,
            sortino=1.5,
            calmar=2.0,
            profit_factor=1.8,
            max_consecutive_losses=4,
            payoff_ratio=1.3,
        ),
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.backtest_run import PgBacktestRunRepository

    return PgBacktestRunRepository(connection=db_connection)


async def test_insert_and_get_by_hash(repo: Any, clean_tables: None) -> None:
    record = _make_record("hash-abc")
    await repo.insert(record)

    rows = await repo.get_by_config_hash("hash-abc")
    assert len(rows) == 1
    assert rows[0].config_hash == "hash-abc"
    assert rows[0].summary.trades == 100


async def test_get_by_hash_not_found(repo: Any, clean_tables: None) -> None:
    rows = await repo.get_by_config_hash("nonexistent")
    assert rows == ()


async def test_insert_returns_positive_id(repo: Any, clean_tables: None) -> None:
    record = _make_record("hash-xyz")
    row_id = await repo.insert(record)
    assert row_id > 0

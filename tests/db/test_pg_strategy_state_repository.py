"""Integration tests for PgStrategyStateRepository."""

from typing import Any

import pytest

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.strategy_state import PgStrategyStateRepository

    return PgStrategyStateRepository(connection=db_connection)


async def test_upsert_and_get(repo: Any, clean_tables: None) -> None:
    await repo.upsert_state(
        "BTCUSDC", "momentum", "v1", {"last_signal": "buy"}, target_id="btcusdc", venue="binance", product="usd_c"
    )

    state = await repo.get_state("BTCUSDC", "momentum", "v1")
    assert state is not None
    assert state["last_signal"] == "buy"


async def test_get_returns_none(repo: Any, clean_tables: None) -> None:
    state = await repo.get_state("ETHUSDC", "mean_reversion", "v1")
    assert state is None


async def test_upsert_overwrites(repo: Any, clean_tables: None) -> None:
    await repo.upsert_state(
        "BTCUSDC", "momentum", "v1", {"count": 1}, target_id="btcusdc", venue="binance", product="usd_c"
    )
    await repo.upsert_state(
        "BTCUSDC", "momentum", "v1", {"count": 99}, target_id="btcusdc", venue="binance", product="usd_c"
    )

    state = await repo.get_state("BTCUSDC", "momentum", "v1")
    assert state is not None
    assert state["count"] == 99

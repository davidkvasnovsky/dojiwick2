"""Integration tests for PgPairStateRepository."""

from typing import Any

import pytest

from dojiwick.domain.models.entities.pair_state import PairTradingState

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.pair_state import PgPairStateRepository

    return PgPairStateRepository(connection=db_connection)


async def test_upsert_and_get(repo: Any, clean_tables: None) -> None:
    state = PairTradingState(
        pair="BTC/USDC", target_id="BTC/USDC", venue="binance", product="usd_c", wins=5, losses=2, consecutive_losses=1
    )
    await repo.upsert(state)
    loaded = await repo.get_state("BTC/USDC")
    assert loaded is not None
    assert loaded.wins == 5
    assert loaded.losses == 2
    assert loaded.venue == "binance"
    assert loaded.product == "usd_c"


async def test_get_all(repo: Any, clean_tables: None) -> None:
    await repo.upsert(PairTradingState(pair="BTC/USDC", target_id="BTC/USDC", venue="binance", product="usd_c"))
    await repo.upsert(PairTradingState(pair="ETH/USDC", target_id="ETH/USDC", venue="binance", product="usd_c"))
    all_states = await repo.get_all()
    pairs = {s.pair for s in all_states}
    assert "BTC/USDC" in pairs
    assert "ETH/USDC" in pairs


async def test_get_returns_none_for_unknown(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_state("UNKNOWN/PAIR")
    assert loaded is None

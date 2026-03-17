"""Integration tests for PgBotStateRepository."""

from typing import Any

import pytest

from dojiwick.domain.models.entities.bot_state import BotState

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.bot_state import PgBotStateRepository

    return PgBotStateRepository(connection=db_connection)


async def test_upsert_and_get(repo: Any, clean_tables: None) -> None:
    state = BotState(consecutive_errors=3, daily_trade_count=5)
    await repo.update_state(state)
    loaded = await repo.get_state()
    assert loaded is not None
    assert loaded.consecutive_errors == 3
    assert loaded.daily_trade_count == 5


async def test_get_returns_none_when_empty(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_state()
    assert loaded is None

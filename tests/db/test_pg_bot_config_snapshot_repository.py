"""Integration tests for PgBotConfigSnapshotRepository."""

from typing import Any

import pytest

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.bot_config_snapshot import PgBotConfigSnapshotRepository

    return PgBotConfigSnapshotRepository(connection=db_connection)


async def test_record_and_get_latest(repo: Any, clean_tables: None) -> None:
    await repo.record_snapshot("hash-1", '{"key": "value"}')

    result = await repo.get_latest()
    assert result is not None
    config_hash, config_json = result
    assert config_hash == "hash-1"
    assert "key" in config_json


async def test_get_latest_none_when_empty(repo: Any, clean_tables: None) -> None:
    result = await repo.get_latest()
    assert result is None


async def test_latest_returns_most_recent(repo: Any, clean_tables: None) -> None:
    await repo.record_snapshot("hash-old", '{"v": 1}')
    await repo.record_snapshot("hash-new", '{"v": 2}')

    result = await repo.get_latest()
    assert result is not None
    assert result[0] == "hash-new"

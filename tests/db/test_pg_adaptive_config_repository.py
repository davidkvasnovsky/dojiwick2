"""Integration tests for PgAdaptiveConfigRepository."""

from typing import Any

import pytest
import pytest_asyncio

from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def seed_configs(db_cursor: DbCursor, clean_tables: None) -> None:
    """Insert test adaptive configs."""
    await db_cursor.execute("""
        INSERT INTO adaptive_configs (config_idx, params_json)
        VALUES (0, '{"k": 1, "name": "fast"}'),
               (1, '{"k": 2, "name": "slow"}')
    """)


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.adaptive_config import PgAdaptiveConfigRepository

    return PgAdaptiveConfigRepository(connection=db_connection)


async def test_get_config(repo: Any, seed_configs: None) -> None:
    result = await repo.get_config(0)
    assert result is not None
    assert result["k"] == 1
    assert result["name"] == "fast"


async def test_get_config_not_found(repo: Any, clean_tables: None) -> None:
    result = await repo.get_config(999)
    assert result is None


async def test_get_all_configs(repo: Any, seed_configs: None) -> None:
    result = await repo.get_all_configs()
    assert len(result) == 2
    assert result[0][0] == 0
    assert result[1][0] == 1
    assert result[0][1]["name"] == "fast"
    assert result[1][1]["name"] == "slow"

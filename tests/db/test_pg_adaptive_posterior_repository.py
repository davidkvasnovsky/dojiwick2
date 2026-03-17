"""Integration tests for PgAdaptivePosteriorRepository."""

from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptivePosterior
from dojiwick.infrastructure.postgres.connection import DbCursor

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def seed_config(db_cursor: DbCursor, clean_tables: None) -> None:
    """Insert adaptive_configs row for FK."""
    await db_cursor.execute("""
        INSERT INTO adaptive_configs (config_idx, params_json)
        VALUES (0, '{"k": 1}')
        ON CONFLICT DO NOTHING
    """)


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.adaptive_posterior import PgAdaptivePosteriorRepository

    return PgAdaptivePosteriorRepository(connection=db_connection)


def _make_posterior(regime_idx: int = 1, config_idx: int = 0) -> AdaptivePosterior:
    return AdaptivePosterior(
        arm=AdaptiveArmKey(regime_idx=regime_idx, config_idx=config_idx),
        alpha=2.0,
        beta=3.0,
        n_updates=5,
        last_decay_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


async def test_upsert_and_get_posteriors(repo: Any, seed_config: None) -> None:
    posterior = _make_posterior()
    await repo.upsert_posterior(posterior)

    loaded = await repo.get_posteriors(1)
    assert len(loaded) == 1
    assert loaded[0].alpha == 2.0
    assert loaded[0].beta == 3.0
    assert loaded[0].n_updates == 5
    assert loaded[0].last_decay_at is not None


async def test_upsert_idempotent(repo: Any, seed_config: None) -> None:
    """Upsert same key twice — second call should update values."""
    await repo.upsert_posterior(_make_posterior())

    updated = AdaptivePosterior(
        arm=AdaptiveArmKey(regime_idx=1, config_idx=0),
        alpha=5.0,
        beta=8.0,
        n_updates=10,
        last_decay_at=datetime(2025, 6, 1, tzinfo=UTC),
    )
    await repo.upsert_posterior(updated)

    loaded = await repo.get_posteriors(1)
    assert len(loaded) == 1
    assert loaded[0].alpha == 5.0
    assert loaded[0].beta == 8.0
    assert loaded[0].n_updates == 10


async def test_get_posteriors_empty(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_posteriors(999)
    assert loaded == ()

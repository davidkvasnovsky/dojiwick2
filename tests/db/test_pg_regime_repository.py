"""Integration tests for PgRegimeRepository."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest

from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.regime import PgRegimeRepository

    return PgRegimeRepository(connection=db_connection)


async def test_insert_batch(repo: Any, db_cursor: Any) -> None:
    pairs = ("BTC/USDC", "ETH/USDC")
    regimes = BatchRegimeProfile(
        coarse_state=np.array([1, 3], dtype=np.int64),
        confidence=np.array([0.85, 0.70], dtype=np.float64),
        valid_mask=np.array([True, True], dtype=np.bool_),
    )
    await repo.insert_batch(pairs, datetime.now(UTC), regimes, target_ids=pairs, venue="binance", product="usd_c")
    await db_cursor.execute("SELECT COUNT(*) FROM regime_observations")
    row = await db_cursor.fetchone()
    assert row is not None
    assert row[0] >= 2

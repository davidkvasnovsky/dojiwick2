"""Batch model validation tests."""

from datetime import UTC, datetime

import numpy as np
import pytest

from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import BatchMarketSnapshot


def test_batch_market_snapshot_validates_shape() -> None:
    with pytest.raises(ValueError, match="indicators row count mismatch"):
        BatchMarketSnapshot(
            pairs=("BTC/USDC",),
            observed_at=datetime.now(UTC),
            price=np.array([100.0], dtype=np.float64),
            indicators=np.zeros((2, INDICATOR_COUNT), dtype=np.float64),
        )

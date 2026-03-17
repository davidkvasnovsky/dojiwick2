"""Integration tests for PgAdaptiveCalibrationRepository."""

from typing import Any

import pytest

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveCalibrationMetrics

pytestmark = pytest.mark.db


def _make_metrics(regime_idx: int) -> AdaptiveCalibrationMetrics:
    return AdaptiveCalibrationMetrics(
        arm=AdaptiveArmKey(regime_idx=regime_idx, config_idx=0),
        expected_reward=0.6,
        selection_count=50,
        empirical_reward=0.55,
        posterior_variance=0.02,
        calibration_gap=0.05,
    )


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.adaptive_calibration import PgAdaptiveCalibrationRepository

    return PgAdaptiveCalibrationRepository(connection=db_connection)


async def test_record_and_get_by_regime(repo: Any, clean_tables: None) -> None:
    metrics = _make_metrics(regime_idx=1)
    await repo.record_metric(metrics)

    results = await repo.get_latest_by_regime(1)
    assert len(results) == 1
    assert results[0].expected_reward == 0.6
    assert results[0].selection_count == 50


async def test_get_by_regime_empty(repo: Any, clean_tables: None) -> None:
    results = await repo.get_latest_by_regime(999)
    assert results == ()

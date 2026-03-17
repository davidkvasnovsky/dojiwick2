"""PostgreSQL adaptive calibration repository."""

from dataclasses import dataclass

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveCalibrationMetrics

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO adaptive_calibration_metrics (regime_idx, metric_name, metric_value, computed_at)
VALUES (%s, %s, %s, now())
"""

_SELECT_BY_REGIME_SQL = """
SELECT regime_idx, metric_name, metric_value
FROM adaptive_calibration_metrics
WHERE regime_idx = %s
ORDER BY computed_at DESC
"""


@dataclass(slots=True)
class PgAdaptiveCalibrationRepository:
    """Persists adaptive calibration metrics into PostgreSQL."""

    connection: DbConnection

    async def record_metric(self, metric: AdaptiveCalibrationMetrics) -> None:
        """Persist a calibration metric."""
        metrics_to_insert = [
            (metric.arm.regime_idx, "expected_reward", metric.expected_reward),
            (metric.arm.regime_idx, "selection_count", float(metric.selection_count)),
            (metric.arm.regime_idx, "empirical_reward", metric.empirical_reward),
            (metric.arm.regime_idx, "posterior_variance", metric.posterior_variance),
            (metric.arm.regime_idx, "calibration_gap", metric.calibration_gap),
        ]
        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_INSERT_SQL, metrics_to_insert)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record calibration metrics: {exc}") from exc

    async def get_latest_by_regime(self, regime_idx: int) -> tuple[AdaptiveCalibrationMetrics, ...]:
        """Return the latest calibration metrics for a given regime."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_REGIME_SQL, (regime_idx,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get calibration metrics: {exc}") from exc
        metrics_by_arm: dict[int, dict[str, float]] = {}
        for row in rows:
            r_idx = int(str(row[0]))
            name = str(row[1])
            value = float(str(row[2]))
            if r_idx not in metrics_by_arm:
                metrics_by_arm[r_idx] = {}
            if name not in metrics_by_arm[r_idx]:
                metrics_by_arm[r_idx][name] = value
        results: list[AdaptiveCalibrationMetrics] = []
        for r_idx, vals in metrics_by_arm.items():
            if len(vals) >= 5:
                results.append(
                    AdaptiveCalibrationMetrics(
                        arm=AdaptiveArmKey(regime_idx=r_idx, config_idx=0),
                        expected_reward=vals.get("expected_reward", 0.0),
                        selection_count=int(vals.get("selection_count", 0)),
                        empirical_reward=vals.get("empirical_reward", 0.0),
                        posterior_variance=vals.get("posterior_variance", 0.0),
                        calibration_gap=vals.get("calibration_gap", 0.0),
                    )
                )
        return tuple(results)

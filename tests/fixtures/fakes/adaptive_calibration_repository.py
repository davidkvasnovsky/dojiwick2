"""Fake adaptive calibration repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.adaptive import AdaptiveCalibrationMetrics


@dataclass(slots=True)
class FakeAdaptiveCalibrationRepository:
    """In-memory adaptive calibration repository for test assertions."""

    metrics: list[AdaptiveCalibrationMetrics] = field(default_factory=list)

    async def record_metric(self, metric: AdaptiveCalibrationMetrics) -> None:
        self.metrics.append(metric)

    async def get_latest_by_regime(self, regime_idx: int) -> tuple[AdaptiveCalibrationMetrics, ...]:
        return tuple(m for m in self.metrics if m.arm.regime_idx == regime_idx)

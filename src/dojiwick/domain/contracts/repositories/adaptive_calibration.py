"""Adaptive calibration repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveCalibrationMetrics


class AdaptiveCalibrationRepositoryPort(Protocol):
    """Adaptive calibration metrics persistence."""

    async def record_metric(self, metric: AdaptiveCalibrationMetrics) -> None:
        """Persist a calibration metric."""
        ...

    async def get_latest_by_regime(self, regime_idx: int) -> tuple[AdaptiveCalibrationMetrics, ...]:
        """Return the latest calibration metrics for a given regime."""
        ...

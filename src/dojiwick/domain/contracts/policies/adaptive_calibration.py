"""Adaptive calibration policy protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveCalibrationMetrics, AdaptivePosterior


class AdaptiveCalibrationPolicyPort(Protocol):
    """Evaluates and recalibrates posteriors based on calibration metrics."""

    async def evaluate(
        self,
        posteriors: tuple[AdaptivePosterior, ...],
        metrics: tuple[AdaptiveCalibrationMetrics, ...],
    ) -> tuple[AdaptivePosterior, ...]:
        """Return recalibrated posteriors."""
        ...

"""Adaptive selection policy protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptivePosterior


class AdaptiveSelectionPolicyPort(Protocol):
    """Selects a bandit arm given the current posteriors."""

    async def select(self, regime_idx: int, posteriors: tuple[AdaptivePosterior, ...]) -> AdaptiveArmKey:
        """Return the arm key chosen for the given regime and posteriors."""
        ...

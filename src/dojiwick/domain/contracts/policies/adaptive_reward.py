"""Adaptive reward policy protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveOutcomeEvent, AdaptiveSelectionEvent
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord, PositionLeg


class AdaptiveRewardPolicyPort(Protocol):
    """Computes a reward signal from a closed position leg."""

    async def compute_reward(
        self,
        leg: PositionLeg,
        close_event: PositionEventRecord,
        selection: AdaptiveSelectionEvent,
    ) -> AdaptiveOutcomeEvent:
        """Return an outcome event with the computed reward."""
        ...

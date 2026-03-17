"""Adaptive outcome repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveOutcomeEvent


class AdaptiveOutcomeRepositoryPort(Protocol):
    """Adaptive outcome event persistence."""

    async def record_outcome(self, event: AdaptiveOutcomeEvent) -> None:
        """Persist an outcome event."""
        ...

    async def get_outcome(self, position_leg_id: int) -> AdaptiveOutcomeEvent | None:
        """Return the outcome event for a position leg, or None if absent."""
        ...

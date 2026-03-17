"""Adaptive selection repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptiveSelectionEvent


class AdaptiveSelectionRepositoryPort(Protocol):
    """Adaptive arm selection event persistence."""

    async def record_selection(self, event: AdaptiveSelectionEvent) -> None:
        """Persist an arm selection event."""
        ...

    async def get_selection(self, position_leg_id: int) -> AdaptiveSelectionEvent | None:
        """Return the selection event for a position leg, or None if absent."""
        ...

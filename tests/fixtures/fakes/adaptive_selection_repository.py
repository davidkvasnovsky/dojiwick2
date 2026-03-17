"""Adaptive selection repository test double."""

from dojiwick.domain.models.value_objects.adaptive import AdaptiveSelectionEvent


class FakeAdaptiveSelectionRepository:
    """In-memory fake for AdaptiveSelectionRepositoryPort."""

    def __init__(self) -> None:
        self._selections: dict[int, AdaptiveSelectionEvent] = {}

    async def record_selection(self, event: AdaptiveSelectionEvent) -> None:
        self._selections[event.position_leg_id] = event

    async def get_selection(self, position_leg_id: int) -> AdaptiveSelectionEvent | None:
        return self._selections.get(position_leg_id)

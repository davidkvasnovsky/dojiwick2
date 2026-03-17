"""Fake adaptive outcome repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.adaptive import AdaptiveOutcomeEvent


@dataclass(slots=True)
class FakeAdaptiveOutcomeRepository:
    """In-memory adaptive outcome repository for test assertions."""

    outcomes: list[AdaptiveOutcomeEvent] = field(default_factory=list)

    async def record_outcome(self, event: AdaptiveOutcomeEvent) -> None:
        self.outcomes.append(event)

    async def get_outcome(self, position_leg_id: int) -> AdaptiveOutcomeEvent | None:
        for o in self.outcomes:
            if o.position_leg_id == position_leg_id:
                return o
        return None

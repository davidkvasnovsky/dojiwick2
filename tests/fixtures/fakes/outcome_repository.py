"""Outcome repository test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome


class FailingOutcomeRepo:
    """Raises on append_outcomes."""

    async def append_outcomes(self, outcomes: tuple[DecisionOutcome, ...], *, venue: str, product: str) -> None:
        del outcomes, venue, product
        raise RuntimeError("persistence failure")


@dataclass(slots=True)
class CapturingOutcomeRepo:
    """Stores all appended outcomes for assertion."""

    outcomes: list[DecisionOutcome] = field(default_factory=list)

    async def append_outcomes(self, outcomes: tuple[DecisionOutcome, ...], *, venue: str, product: str) -> None:
        del venue, product
        self.outcomes.extend(outcomes)

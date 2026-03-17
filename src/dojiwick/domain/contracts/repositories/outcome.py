"""Outcome repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome


class OutcomeRepositoryPort(Protocol):
    """Persists per-pair outcomes."""

    async def append_outcomes(self, outcomes: tuple[DecisionOutcome, ...], *, venue: str, product: str) -> None:
        """Persist one batch of outcomes."""
        ...

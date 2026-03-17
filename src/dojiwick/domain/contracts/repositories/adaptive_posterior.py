"""Adaptive posterior repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.adaptive import AdaptivePosterior


class AdaptivePosteriorRepositoryPort(Protocol):
    """Adaptive posterior persistence for bandit arms."""

    async def get_posteriors(self, regime_idx: int) -> tuple[AdaptivePosterior, ...]:
        """Return all posteriors for a given regime."""
        ...

    async def upsert_posterior(self, posterior: AdaptivePosterior) -> None:
        """Insert or update a posterior."""
        ...

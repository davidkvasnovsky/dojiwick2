"""Adaptive posterior repository test double."""

from dojiwick.domain.models.value_objects.adaptive import AdaptivePosterior


class FakeAdaptivePosteriorRepository:
    """In-memory fake for AdaptivePosteriorRepositoryPort."""

    def __init__(self) -> None:
        self._posteriors: dict[tuple[int, int], AdaptivePosterior] = {}

    async def get_posteriors(self, regime_idx: int) -> tuple[AdaptivePosterior, ...]:
        return tuple(p for (r, _), p in self._posteriors.items() if r == regime_idx)

    async def upsert_posterior(self, posterior: AdaptivePosterior) -> None:
        key = (posterior.arm.regime_idx, posterior.arm.config_idx)
        self._posteriors[key] = posterior

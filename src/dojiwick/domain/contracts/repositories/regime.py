"""Regime observation repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile


class RegimeRepositoryPort(Protocol):
    """Persists classified regimes for observability."""

    async def insert_batch(
        self,
        pairs: tuple[str, ...],
        observed_at: datetime,
        regimes: BatchRegimeProfile,
        *,
        target_ids: tuple[str, ...],
        venue: str,
        product: str,
    ) -> None:
        """Persist one classified regime batch."""
        ...

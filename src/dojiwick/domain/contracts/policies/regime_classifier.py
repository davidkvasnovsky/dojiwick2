"""AI regime classifier protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
)


class AIRegimeClassifierPort(Protocol):
    """Optional AI regime classifier for ensemble regime detection."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        """Return AI regime classification given the deterministic baseline."""
        ...

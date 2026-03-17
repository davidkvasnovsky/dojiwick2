"""AI veto protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    BatchVetoDecision,
)


class VetoServicePort(Protocol):
    """Optional AI veto service for candidate filtering."""

    async def evaluate_batch(self, context: BatchDecisionContext, candidates: BatchTradeCandidate) -> BatchVetoDecision:
        """Return per-pair approval mask for deterministic candidates."""
        ...

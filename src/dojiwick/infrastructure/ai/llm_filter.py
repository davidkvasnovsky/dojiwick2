"""Null veto service that approves all candidates."""

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    BatchVetoDecision,
)


class NullVetoService:
    """Approves every row and preserves deterministic authority semantics."""

    async def evaluate_batch(self, context: BatchDecisionContext, candidates: BatchTradeCandidate) -> BatchVetoDecision:
        """Return approval for each candidate row."""

        del candidates
        size = context.size
        return BatchVetoDecision(
            approved_mask=np.ones(size, dtype=np.bool_),
            reason_codes=tuple("veto_approved" for _ in range(size)),
        )

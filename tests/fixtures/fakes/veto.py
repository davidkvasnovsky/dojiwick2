"""Veto service test doubles."""

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    BatchVetoDecision,
)


class NullVeto:
    """Approves everything."""

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        del candidates
        size = context.size
        return BatchVetoDecision(
            approved_mask=np.ones(size, dtype=np.bool_),
            reason_codes=tuple("approved" for _ in range(size)),
        )


class RejectFirstVeto:
    """Rejects the first pair, approves the rest."""

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        del candidates
        approved = np.ones(context.size, dtype=np.bool_)
        approved[0] = False
        return BatchVetoDecision(
            approved_mask=approved,
            reason_codes=("rejected",) + tuple("ok" for _ in range(context.size - 1)),
        )


class RejectAllVeto:
    """Rejects every pair."""

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        del candidates
        size = context.size
        return BatchVetoDecision(
            approved_mask=np.zeros(size, dtype=np.bool_),
            reason_codes=tuple("rejected" for _ in range(size)),
        )


class FailVeto:
    """Raises the specified exception on evaluate."""

    def __init__(self, raises: type[Exception] = RuntimeError) -> None:
        self._raises = raises

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        del context, candidates
        raise self._raises("veto service failure")


class TimeoutVeto:
    """Raises TimeoutError on evaluate."""

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        del context, candidates
        raise TimeoutError("veto timed out")

"""AI regime classifier test doubles."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
)


class AgreeingClassifier:
    """Echoes the deterministic regime — full agreement."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context
        return deterministic_regime


class DisagreeingClassifier:
    """Classifies everything as RANGING — guaranteed disagreement with non-RANGING regimes."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context
        size = len(deterministic_regime.coarse_state)
        return BatchRegimeProfile(
            coarse_state=np.full(size, MarketState.RANGING.value, dtype=np.int64),
            confidence=np.ones(size, dtype=np.float64),
            valid_mask=np.ones(size, dtype=np.bool_),
        )


class PartiallyValidClassifier:
    """Valid for the first pair only; rest are invalid."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context
        size = len(deterministic_regime.coarse_state)
        valid = np.zeros(size, dtype=np.bool_)
        valid[0] = True
        return BatchRegimeProfile(
            coarse_state=deterministic_regime.coarse_state.copy(),
            confidence=deterministic_regime.confidence.copy(),
            valid_mask=valid,
        )


class FailingClassifier:
    """Raises a configurable exception on classify."""

    def __init__(self, raises: type[Exception] = RuntimeError) -> None:
        self._raises = raises

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context, deterministic_regime
        raise self._raises("regime classifier failure")


class TimeoutClassifier:
    """Raises TimeoutError on classify."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context, deterministic_regime
        raise TimeoutError("regime classifier timed out")

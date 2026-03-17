"""Null regime classifier that echoes the deterministic regime unchanged."""

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
)


class NullRegimeClassifier:
    """Echoes the deterministic regime — equivalent to full agreement."""

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        del context
        return deterministic_regime

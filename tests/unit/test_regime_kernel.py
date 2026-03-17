"""Regime kernel tests."""

import numpy as np

from fixtures.factories.infrastructure import default_settings
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from dojiwick.compute.kernels.regime.classify import classify_regime_batch


def test_classify_regime_batch_detects_trending_up(sample_context: BatchDecisionContext) -> None:
    settings = default_settings()
    result = classify_regime_batch(sample_context.market, settings.regime.params)

    assert np.all(result.coarse_state == 1)
    assert np.all(result.valid_mask)
    assert np.all(result.confidence > 0.0)


def test_trending_confidence_exceeds_085(sample_context: BatchDecisionContext) -> None:
    """After vol_component fix, trending confidence can exceed 0.85."""
    settings = default_settings()
    result = classify_regime_batch(sample_context.market, settings.regime.params)

    assert np.all(result.coarse_state == 1), "expected TRENDING_UP"
    assert np.all(result.confidence > 0.85), f"trending confidence should exceed 0.85, got {result.confidence}"

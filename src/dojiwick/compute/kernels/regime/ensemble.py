"""Regime ensemble kernel — combines deterministic and AI regime classifications."""

from dojiwick.compute.kernels.math import clamp01
from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile


def combine_regime_ensemble(
    deterministic: BatchRegimeProfile,
    ai: BatchRegimeProfile,
    boost: float,
    penalty: float,
) -> BatchRegimeProfile:
    """Combine deterministic and AI regime profiles into an ensemble result.

    Rules:
    - ``coarse_state`` is **always** the deterministic state (AI never overrides).
    - Where AI and deterministic agree: ``confidence = clamp01(det_confidence * boost)``.
    - Where they disagree: ``confidence = clamp01(det_confidence * penalty)``.
    - Where AI is invalid (``~ai.valid_mask``): deterministic confidence unchanged.
    """
    agree = ai.coarse_state == deterministic.coarse_state
    ai_valid = ai.valid_mask

    confidence = deterministic.confidence.copy()

    boosted = ai_valid & agree
    penalised = ai_valid & ~agree

    confidence[boosted] = clamp01(deterministic.confidence[boosted] * boost)
    confidence[penalised] = clamp01(deterministic.confidence[penalised] * penalty)

    return BatchRegimeProfile(
        coarse_state=deterministic.coarse_state.copy(),
        confidence=confidence,
        valid_mask=deterministic.valid_mask.copy(),
    )

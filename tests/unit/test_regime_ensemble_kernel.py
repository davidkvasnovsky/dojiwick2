"""Unit tests for the regime ensemble kernel."""

from collections.abc import Sequence

import numpy as np
from numpy.testing import assert_array_equal

from dojiwick.compute.kernels.regime.ensemble import combine_regime_ensemble
from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile

BOOST = 1.25
PENALTY = 0.6


def _regime(
    states: Sequence[int],
    confidences: Sequence[float],
    valid: Sequence[bool] | None = None,
) -> BatchRegimeProfile:
    size = len(states)
    return BatchRegimeProfile(
        coarse_state=np.array(states, dtype=np.int64),
        confidence=np.array(confidences, dtype=np.float64),
        valid_mask=np.array(valid if valid is not None else [True] * size, dtype=np.bool_),
    )


class TestAgreement:
    def test_agreement_boosts_confidence(self) -> None:
        det = _regime([MarketState.TRENDING_UP], [0.8])
        ai = _regime([MarketState.TRENDING_UP], [0.9])
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert result.confidence[0] == min(0.8 * BOOST, 1.0)
        assert_array_equal(result.coarse_state, det.coarse_state)

    def test_agreement_clamps_to_one(self) -> None:
        det = _regime([MarketState.TRENDING_UP], [0.95])
        ai = _regime([MarketState.TRENDING_UP], [0.9])
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert result.confidence[0] == 1.0

    def test_full_batch_agreement(self) -> None:
        states = [MarketState.TRENDING_UP, MarketState.RANGING, MarketState.VOLATILE]
        det = _regime(states, [0.7, 0.6, 0.5])
        ai = _regime(states, [0.8, 0.7, 0.6])
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        expected = np.clip(np.array([0.7, 0.6, 0.5]) * BOOST, 0.0, 1.0)
        np.testing.assert_allclose(result.confidence, expected)


class TestDisagreement:
    def test_disagreement_penalises_confidence(self) -> None:
        det = _regime([MarketState.TRENDING_UP], [0.8])
        ai = _regime([MarketState.RANGING], [0.9])
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert result.confidence[0] == 0.8 * PENALTY
        assert_array_equal(result.coarse_state, det.coarse_state)

    def test_disagreement_clamps_to_zero(self) -> None:
        det = _regime([MarketState.TRENDING_UP], [0.0])
        ai = _regime([MarketState.RANGING], [0.9])
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert result.confidence[0] == 0.0


class TestDeterministicStateNeverOverridden:
    def test_coarse_state_always_deterministic(self) -> None:
        det = _regime(
            [MarketState.TRENDING_UP, MarketState.VOLATILE],
            [0.8, 0.7],
        )
        ai = _regime(
            [MarketState.RANGING, MarketState.TRENDING_DOWN],
            [0.9, 0.9],
        )
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert_array_equal(result.coarse_state, det.coarse_state)


class TestAIInvalid:
    def test_invalid_ai_leaves_confidence_unchanged(self) -> None:
        det = _regime([MarketState.TRENDING_UP, MarketState.RANGING], [0.8, 0.6])
        ai = _regime(
            [MarketState.RANGING, MarketState.RANGING],
            [0.5, 0.5],
            valid=[False, False],
        )
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        np.testing.assert_allclose(result.confidence, det.confidence)

    def test_partially_valid_ai(self) -> None:
        det = _regime(
            [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN],
            [0.8, 0.7],
        )
        ai = _regime(
            [MarketState.TRENDING_UP, MarketState.RANGING],
            [0.9, 0.9],
            valid=[True, False],
        )
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        # First pair: agrees, boosted
        assert result.confidence[0] == min(0.8 * BOOST, 1.0)
        # Second pair: AI invalid, unchanged
        assert result.confidence[1] == 0.7


class TestMixed:
    def test_mixed_agree_disagree_invalid(self) -> None:
        det = _regime(
            [MarketState.TRENDING_UP, MarketState.RANGING, MarketState.VOLATILE],
            [0.8, 0.6, 0.5],
        )
        ai = _regime(
            [MarketState.TRENDING_UP, MarketState.VOLATILE, MarketState.VOLATILE],
            [0.9, 0.9, 0.9],
            valid=[True, True, False],
        )
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        # Pair 0: agree → boosted
        assert result.confidence[0] == min(0.8 * BOOST, 1.0)
        # Pair 1: disagree → penalised
        assert result.confidence[1] == 0.6 * PENALTY
        # Pair 2: AI invalid → unchanged
        assert result.confidence[2] == 0.5

    def test_valid_mask_always_from_deterministic(self) -> None:
        det = _regime(
            [MarketState.TRENDING_UP, MarketState.RANGING],
            [0.8, 0.0],
            valid=[True, False],
        )
        ai = _regime(
            [MarketState.TRENDING_UP, MarketState.RANGING],
            [0.9, 0.9],
        )
        result = combine_regime_ensemble(det, ai, BOOST, PENALTY)

        assert_array_equal(result.valid_mask, det.valid_mask)

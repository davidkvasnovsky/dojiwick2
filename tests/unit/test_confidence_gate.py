"""Tests for the confidence gate pre-filter."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.infrastructure.ai.confidence_gate import compute_llm_review_mask
from fixtures.factories.compute import RegimeProfileBuilder, TradeCandidateBuilder


class TestConfidenceGate:
    def test_high_confidence_trending_up_skips_review(self) -> None:
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_UP]).with_confidences([0.95]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert not mask[0]

    def test_high_confidence_trending_down_skips_review(self) -> None:
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_DOWN]).with_confidences([0.90]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert not mask[0]

    def test_volatile_always_reviewed(self) -> None:
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.VOLATILE]).with_confidences([0.99]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert mask[0]

    def test_ranging_low_confidence_reviewed(self) -> None:
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.RANGING]).with_confidences([0.60]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert mask[0]

    def test_invalid_candidate_skipped(self) -> None:
        candidates = TradeCandidateBuilder(1).with_valid_mask([False]).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.VOLATILE]).with_confidences([0.99]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert not mask[0]

    def test_correct_shape(self) -> None:
        candidates = TradeCandidateBuilder(3).build()
        regimes = (
            RegimeProfileBuilder(3)
            .with_states([MarketState.TRENDING_UP, MarketState.VOLATILE, MarketState.RANGING])
            .with_confidences([0.95, 0.80, 0.50])
            .build()
        )
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert mask.shape == (3,)
        assert mask.dtype == np.bool_
        assert not mask[0]  # high-confidence trending — skip
        assert mask[1]  # volatile — always review
        assert mask[2]  # ranging low-confidence — review

    def test_trending_below_threshold_reviewed(self) -> None:
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_UP]).with_confidences([0.70]).build()
        mask = compute_llm_review_mask(candidates, regimes, confidence_threshold=0.85)
        assert mask[0]

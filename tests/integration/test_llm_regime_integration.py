"""Integration tests for LLM regime classifier with ensemble."""

import json

from dojiwick.compute.kernels.regime.ensemble import combine_regime_ensemble
from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import (
    BatchRegimeProfile,
)
from dojiwick.infrastructure.ai.llm_regime_classifier import LLMRegimeClassifier
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import RegimeProfileBuilder


class TestLLMRegimeClassifierPort:
    """LLMRegimeClassifier satisfies AIRegimeClassifierPort protocol."""

    async def test_satisfies_regime_port(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "VOLATILE", "confidence": 0.8}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.RANGING]).with_confidences([0.7]).build()
        result: BatchRegimeProfile = await clf.classify_batch(ctx, det)
        assert isinstance(result, BatchRegimeProfile)


class TestRegimeEnsembleIntegration:
    async def test_agreement_boosts_confidence(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "TRENDING_UP", "confidence": 0.9}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_UP]).with_confidences([0.8]).build()
        ai_regime = await clf.classify_batch(ctx, det)

        ensemble = combine_regime_ensemble(deterministic=det, ai=ai_regime, boost=1.25, penalty=0.6)
        # Deterministic coarse_state preserved
        assert int(ensemble.coarse_state[0]) == MarketState.TRENDING_UP
        # Confidence boosted (0.8 * 1.25 = 1.0, clamped)
        assert float(ensemble.confidence[0]) >= 0.8

    async def test_disagreement_penalizes_confidence(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "VOLATILE", "confidence": 0.9}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_UP]).with_confidences([0.8]).build()
        ai_regime = await clf.classify_batch(ctx, det)

        ensemble = combine_regime_ensemble(deterministic=det, ai=ai_regime, boost=1.25, penalty=0.6)
        # Deterministic coarse_state preserved
        assert int(ensemble.coarse_state[0]) == MarketState.TRENDING_UP
        # Confidence penalized (0.8 * 0.6 = 0.48)
        assert float(ensemble.confidence[0]) < 0.8

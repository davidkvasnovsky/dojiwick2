"""Tests for LLM regime classifier."""

import json

from dojiwick.domain.enums import MarketState
from dojiwick.infrastructure.ai.llm_regime_classifier import LLMRegimeClassifier
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import RegimeProfileBuilder


class TestLLMRegimeClassifier:
    async def test_success_returns_ai_regime(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "VOLATILE", "confidence": 0.9}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.RANGING]).with_confidences([0.7]).build()
        result = await clf.classify_batch(ctx, det)
        assert int(result.coarse_state[0]) == MarketState.VOLATILE
        assert float(result.confidence[0]) == 0.9
        assert len(client.calls) == 1

    async def test_parse_failure_echoes_deterministic(self) -> None:
        client = FixedLLMClient("garbage response")
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.TRENDING_UP]).with_confidences([0.85]).build()
        result = await clf.classify_batch(ctx, det)
        assert int(result.coarse_state[0]) == MarketState.TRENDING_UP
        assert float(result.confidence[0]) == 0.85

    async def test_invalid_state_echoes_deterministic(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "UNKNOWN_STATE", "confidence": 0.5}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.RANGING]).with_confidences([0.6]).build()
        result = await clf.classify_batch(ctx, det)
        assert int(result.coarse_state[0]) == MarketState.RANGING
        assert float(result.confidence[0]) == 0.6

    async def test_correct_shape_multi_pair(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "TRENDING_DOWN", "confidence": 0.8}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).build()
        det = (
            RegimeProfileBuilder(2)
            .with_states([MarketState.TRENDING_UP, MarketState.RANGING])
            .with_confidences([0.9, 0.7])
            .build()
        )
        result = await clf.classify_batch(ctx, det)
        assert result.coarse_state.shape == (2,)
        assert result.confidence.shape == (2,)
        assert len(client.calls) == 2

    async def test_invalid_mask_skips_llm(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "VOLATILE", "confidence": 0.9}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = (
            RegimeProfileBuilder(1)
            .with_states([MarketState.RANGING])
            .with_confidences([0.5])
            .with_valid_mask([False])
            .build()
        )
        result = await clf.classify_batch(ctx, det)
        assert int(result.coarse_state[0]) == MarketState.RANGING
        assert len(client.calls) == 0

    async def test_confidence_clamped(self) -> None:
        client = FixedLLMClient(json.dumps({"state": "VOLATILE", "confidence": 1.5}))
        clf = LLMRegimeClassifier(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det = RegimeProfileBuilder(1).with_states([MarketState.RANGING]).with_confidences([0.5]).build()
        result = await clf.classify_batch(ctx, det)
        assert float(result.confidence[0]) == 1.0

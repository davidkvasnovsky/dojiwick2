"""Tests for LLM response hardening — oversized responses, max_tokens, data markers."""

import json

from dojiwick.domain.reason_codes import AI_VETO_PARSE_ERROR
from dojiwick.infrastructure.ai.llm_regime_classifier import LLMRegimeClassifier
from dojiwick.infrastructure.ai.llm_veto_service import LLMVetoService
from dojiwick.infrastructure.ai.prompts.regime_prompt import build_regime_user_prompt
from dojiwick.infrastructure.ai.prompts.veto_prompt import build_veto_user_prompt
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import RegimeProfileBuilder, TradeCandidateBuilder
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient


class TestLLMAdversarial:
    async def test_veto_oversized_response_auto_approves(self) -> None:
        """A response >2000 chars triggers content-length guard and auto-approves."""
        oversized = "x" * 2001
        client = FixedLLMClient(content=oversized)
        svc = LLMVetoService(llm_client=client, model="test", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()

        result = await svc.evaluate_batch(ctx, candidates)

        assert result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_PARSE_ERROR

    async def test_regime_oversized_response_echoes_deterministic(self) -> None:
        """A response >2000 chars keeps deterministic regime values."""
        oversized = "x" * 2001
        client = FixedLLMClient(content=oversized)
        svc = LLMRegimeClassifier(llm_client=client, model="test", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        det_regime = RegimeProfileBuilder(1).build()

        result = await svc.classify_batch(ctx, det_regime)

        # Deterministic values should be preserved (not overwritten)
        assert int(result.coarse_state[0]) == int(det_regime.coarse_state[0])
        assert float(result.confidence[0]) == float(det_regime.confidence[0])

    async def test_veto_passes_max_tokens(self) -> None:
        """LLMRequest uses the service's max_response_tokens."""
        client = FixedLLMClient(json.dumps({"approved": True, "reason": "approved"}))
        svc = LLMVetoService(llm_client=client, model="test", clock=FixedClock(), max_response_tokens=150)
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()

        await svc.evaluate_batch(ctx, candidates)

        assert len(client.calls) == 1
        assert client.calls[0].max_tokens == 150

    def test_veto_prompt_wraps_in_data_tags(self) -> None:
        """build_veto_user_prompt wraps output in <data>...</data>."""
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert prompt.startswith("<data>")
        assert prompt.endswith("</data>")

    def test_regime_prompt_wraps_in_data_tags(self) -> None:
        """build_regime_user_prompt wraps output in <data>...</data>."""
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        regime = RegimeProfileBuilder(1).build()
        prompt = build_regime_user_prompt(ctx, regime, 0)
        assert prompt.startswith("<data>")
        assert prompt.endswith("</data>")

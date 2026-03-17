"""Tests for LLM veto service."""

import json

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import (
    BatchRegimeProfile,
)
from dojiwick.domain.reason_codes import (
    AI_VETO_APPROVED,
    AI_VETO_BUDGET_EXCEEDED,
    AI_VETO_CONFIDENCE_SKIP,
    AI_VETO_CONFLICTING_REGIME,
    AI_VETO_PARSE_ERROR,
)
from dojiwick.infrastructure.ai.cost_tracker import CostTracker
from dojiwick.infrastructure.ai.llm_veto_service import LLMVetoService
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import TradeCandidateBuilder


class TestLLMVetoService:
    async def test_valid_candidate_calls_llm(self) -> None:
        client = FixedLLMClient(json.dumps({"approved": True, "reason": "approved"}))
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).trending_up().build()
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_APPROVED
        assert len(client.calls) == 1

    async def test_invalid_candidate_auto_approved(self) -> None:
        client = FixedLLMClient()
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).with_valid_mask([False]).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert result.approved_mask[0]
        assert result.reason_codes[0] == "no_candidate"
        assert len(client.calls) == 0

    async def test_block_returns_specific_reason(self) -> None:
        client = FixedLLMClient(json.dumps({"approved": False, "reason": "CONFLICTING_REGIME"}))
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert not result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_CONFLICTING_REGIME

    async def test_parse_failure_approves(self) -> None:
        client = FixedLLMClient("not json at all")
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_PARSE_ERROR

    async def test_budget_exceeded_approves(self) -> None:
        clock = FixedClock()
        tracker = CostTracker(
            daily_budget_usd=0.0001,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        # Burn the budget
        await tracker.record(input_tokens=1_000_000, output_tokens=0)

        client = FixedLLMClient()
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock(), cost_tracker=tracker)
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_BUDGET_EXCEEDED
        assert len(client.calls) == 0

    async def test_confidence_skip_with_regime(self) -> None:
        client = FixedLLMClient()
        svc = LLMVetoService(
            llm_client=client, model="claude-sonnet-4-6", clock=FixedClock(), confidence_threshold=0.85
        )
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        # Attach high-confidence trending regime
        from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext

        enriched = BatchDecisionContext(
            market=ctx.market,
            portfolio=ctx.portfolio,
            regimes=BatchRegimeProfile(
                coarse_state=np.array([MarketState.TRENDING_UP], dtype=np.int64),
                confidence=np.array([0.95], dtype=np.float64),
                valid_mask=np.ones(1, dtype=np.bool_),
            ),
        )
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(enriched, candidates)
        assert result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_CONFIDENCE_SKIP
        assert len(client.calls) == 0

    async def test_correct_shape(self) -> None:
        client = FixedLLMClient(json.dumps({"approved": True, "reason": "approved"}))
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).build()
        candidates = TradeCandidateBuilder(2).with_valid_mask([True, False]).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert result.approved_mask.shape == (2,)
        assert len(result.reason_codes) == 2
        assert result.reason_codes[1] == "no_candidate"

"""Integration tests for LLM veto service with the decision pipeline."""

import json

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import (
    BatchVetoDecision,
)
from dojiwick.domain.reason_codes import AI_VETO_APPROVED, AI_VETO_CONFLICTING_REGIME
from dojiwick.infrastructure.ai.llm_filter import NullVetoService
from dojiwick.infrastructure.ai.llm_veto_service import LLMVetoService
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import TradeCandidateBuilder


class TestLLMVetoServicePort:
    """LLMVetoService satisfies VetoServicePort protocol."""

    async def test_satisfies_veto_port(self) -> None:
        client = FixedLLMClient()
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        # Structural subtype check — if this call works, it satisfies the protocol
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        result: BatchVetoDecision = await svc.evaluate_batch(ctx, candidates)
        assert isinstance(result, BatchVetoDecision)

    async def test_approve_all_pipeline(self) -> None:
        client = FixedLLMClient(json.dumps({"approved": True, "reason": "approved"}))
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).build()
        candidates = TradeCandidateBuilder(2).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert np.all(result.approved_mask)
        assert all(code == AI_VETO_APPROVED for code in result.reason_codes)

    async def test_reject_pipeline(self) -> None:
        client = FixedLLMClient(json.dumps({"approved": False, "reason": "CONFLICTING_REGIME"}))
        svc = LLMVetoService(llm_client=client, model="claude-sonnet-4-6", clock=FixedClock())
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        candidates = TradeCandidateBuilder(1).build()
        result = await svc.evaluate_batch(ctx, candidates)
        assert not result.approved_mask[0]
        assert result.reason_codes[0] == AI_VETO_CONFLICTING_REGIME


class TestNullVetoServiceParity:
    """NullVetoService vs always-approve LLM behave identically for approval mask."""

    async def test_parity(self) -> None:
        null_svc = NullVetoService()
        llm_svc = LLMVetoService(
            llm_client=FixedLLMClient(json.dumps({"approved": True, "reason": "approved"})),
            model="claude-sonnet-4-6",
            clock=FixedClock(),
        )
        ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).build()
        candidates = TradeCandidateBuilder(2).build()

        null_result = await null_svc.evaluate_batch(ctx, candidates)
        llm_result = await llm_svc.evaluate_batch(ctx, candidates)

        np.testing.assert_array_equal(null_result.approved_mask, llm_result.approved_mask)

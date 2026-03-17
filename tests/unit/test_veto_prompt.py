"""Tests for veto prompt builder."""

from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext
from dojiwick.infrastructure.ai.prompts.veto_prompt import build_veto_system_prompt, build_veto_user_prompt
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.compute import RegimeProfileBuilder, TradeCandidateBuilder


def _make_context() -> BatchDecisionContext:
    return ContextBuilder(pairs=("BTC/USDC",)).trending_up().build()


class TestVetoPrompt:
    def test_system_prompt_contains_inverted_default(self) -> None:
        prompt = build_veto_system_prompt()
        assert "DEFAULT ACTION: APPROVE" in prompt
        assert "justify any BLOCK" in prompt

    def test_system_prompt_cached(self) -> None:
        a = build_veto_system_prompt()
        b = build_veto_system_prompt()
        assert a is b

    def test_system_prompt_has_four_block_reasons(self) -> None:
        prompt = build_veto_system_prompt()
        assert "CONFLICTING_REGIME" in prompt
        assert "EXTREME_VOLATILITY" in prompt
        assert "CORRELATION_RISK" in prompt
        assert "STALE_SIGNAL" in prompt

    def test_user_prompt_includes_pair_and_price(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert "BTC/USDC" in prompt
        assert "Price:" in prompt

    def test_user_prompt_includes_indicators(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert "rsi:" in prompt
        assert "adx:" in prompt

    def test_user_prompt_includes_candidate_details(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert "BUY" in prompt
        assert "entry=" in prompt
        assert "stop=" in prompt
        assert "tp=" in prompt

    def test_user_prompt_includes_regime_when_provided(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        regimes = RegimeProfileBuilder(1).with_states([1]).with_confidences([0.9]).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0, regimes=regimes)
        assert "TRENDING_UP" in prompt
        assert "confidence:" in prompt

    def test_user_prompt_without_regime(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert "Regime:" not in prompt

    def test_user_prompt_includes_portfolio(self) -> None:
        ctx = _make_context()
        candidates = TradeCandidateBuilder(1).build()
        prompt = build_veto_user_prompt(ctx, candidates, 0)
        assert "equity=" in prompt or "equity=$" in prompt
        assert "open_positions=" in prompt

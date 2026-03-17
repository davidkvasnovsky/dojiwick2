"""Tests for regime classification prompt builder."""

import numpy as np

from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile
from dojiwick.infrastructure.ai.prompts.regime_prompt import build_regime_system_prompt, build_regime_user_prompt
from fixtures.factories.domain import ContextBuilder


class TestRegimePrompt:
    def test_system_prompt_contains_four_regime_labels(self) -> None:
        prompt = build_regime_system_prompt()
        assert "TRENDING_UP" in prompt
        assert "TRENDING_DOWN" in prompt
        assert "RANGING" in prompt
        assert "VOLATILE" in prompt

    def test_system_prompt_cached(self) -> None:
        a = build_regime_system_prompt()
        b = build_regime_system_prompt()
        assert a is b

    def test_user_prompt_includes_deterministic_baseline(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).trending_up().build()
        det_regime = BatchRegimeProfile(
            coarse_state=np.array([1], dtype=np.int64),
            confidence=np.array([0.85], dtype=np.float64),
            valid_mask=np.ones(1, dtype=np.bool_),
        )
        prompt = build_regime_user_prompt(ctx, det_regime, 0)
        assert "TRENDING_UP" in prompt
        assert "confidence:" in prompt
        assert "Deterministic baseline" in prompt

    def test_user_prompt_includes_pair_and_indicators(self) -> None:
        ctx = ContextBuilder(pairs=("ETH/USDC",)).build()
        det_regime = BatchRegimeProfile(
            coarse_state=np.array([3], dtype=np.int64),
            confidence=np.array([0.70], dtype=np.float64),
            valid_mask=np.ones(1, dtype=np.bool_),
        )
        prompt = build_regime_user_prompt(ctx, det_regime, 0)
        assert "ETH/USDC" in prompt
        assert "rsi:" in prompt
        assert "adx:" in prompt

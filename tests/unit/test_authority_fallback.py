"""Tests for authority attribution with fallback codes."""

import numpy as np

from dojiwick.domain.enums import DecisionAuthority
from dojiwick.domain.models.value_objects.batch_models import BatchVetoDecision
from dojiwick.domain.reason_codes import (
    AI_VETO_APPROVED,
    AI_VETO_BUDGET_EXCEEDED,
    AI_VETO_CONFIDENCE_SKIP,
    AI_VETO_CONFLICTING_REGIME,
    AI_VETO_ERROR,
    AI_VETO_PARSE_ERROR,
)
from dojiwick.application.orchestration.decision_pipeline import _authority  # pyright: ignore[reportPrivateUsage]
from dojiwick.infrastructure.ai.llm_veto_service import LLMVetoService
from fixtures.factories.infrastructure import SettingsBuilder
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.llm_client import FixedLLMClient


def _veto(reason_codes: tuple[str, ...]) -> BatchVetoDecision:
    size = len(reason_codes)
    return BatchVetoDecision(
        approved_mask=np.ones(size, dtype=np.bool_),
        reason_codes=reason_codes,
    )


class TestAuthorityFallback:
    def test_all_non_contributory_is_deterministic_only(self) -> None:
        """When all codes are non-contributory, authority is DETERMINISTIC_ONLY."""
        settings = SettingsBuilder().with_ai_veto().build()
        # Using a real service instance as a non-None sentinel
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto((AI_VETO_ERROR, AI_VETO_PARSE_ERROR, AI_VETO_BUDGET_EXCEEDED))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_ONLY

    def test_confidence_skip_is_non_contributory(self) -> None:
        settings = SettingsBuilder().with_ai_veto().build()
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto((AI_VETO_CONFIDENCE_SKIP,))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_ONLY

    def test_mixed_codes_is_ai_active(self) -> None:
        """When at least one code is genuine, AI is active."""
        settings = SettingsBuilder().with_ai_veto().build()
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto((AI_VETO_APPROVED, AI_VETO_PARSE_ERROR))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_PLUS_AI_VETO

    def test_all_genuine_is_ai_active(self) -> None:
        settings = SettingsBuilder().with_ai_veto().build()
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto((AI_VETO_APPROVED, AI_VETO_CONFLICTING_REGIME))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_PLUS_AI_VETO

    def test_veto_service_none_is_deterministic_only(self) -> None:
        settings = SettingsBuilder().with_ai_veto().build()
        veto = _veto((AI_VETO_APPROVED,))
        result = _authority(settings, None, veto)
        assert result == DecisionAuthority.DETERMINISTIC_ONLY

    def test_no_candidate_is_non_contributory(self) -> None:
        settings = SettingsBuilder().with_ai_veto().build()
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto(("no_candidate", "veto_not_enabled"))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_ONLY

    def test_veto_approved_literal_is_non_contributory(self) -> None:
        settings = SettingsBuilder().with_ai_veto().build()
        veto_service = LLMVetoService(llm_client=FixedLLMClient(), model="claude-sonnet-4-6", clock=FixedClock())
        veto = _veto(("veto_approved",))
        result = _authority(settings, veto_service, veto)
        assert result == DecisionAuthority.DETERMINISTIC_ONLY

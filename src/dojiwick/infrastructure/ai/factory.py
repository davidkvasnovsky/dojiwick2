"""AI service composition factory."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dojiwick.config.schema import AISettings
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort
from dojiwick.domain.contracts.policies.regime_classifier import AIRegimeClassifierPort
from dojiwick.domain.contracts.policies.veto import VetoServicePort

if TYPE_CHECKING:
    from dojiwick.domain.contracts.repositories.model_cost import ModelCostRepositoryPort
    from dojiwick.infrastructure.ai.cost_tracker import CostTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AIServices:
    """Bundle of optional AI services."""

    veto_service: VetoServicePort | None
    regime_classifier: AIRegimeClassifierPort | None
    cost_tracker: "CostTracker | None" = None


def build_ai_services(
    settings: AISettings,
    clock: ClockPort,
    metrics: MetricsSinkPort | None = None,
    cost_repository: "ModelCostRepositoryPort | None" = None,
) -> AIServices:
    """Build AI services from settings.

    Returns ``None`` services (not null implementations) when:
    - AI is disabled
    - API key env var is empty/unset

    This is required for correct authority attribution — ``_authority``
    checks ``veto_service is not None`` to determine AI participation.
    """
    if not settings.enabled:
        log.info("AI services disabled")
        return AIServices(veto_service=None, regime_classifier=None)

    api_key = os.environ.get(settings.api_key_env, "").strip()
    if not api_key:
        log.warning(
            "AI enabled but %s not set — running without AI services",
            settings.api_key_env,
        )
        return AIServices(veto_service=None, regime_classifier=None)

    # Lazy imports — anthropic SDK is optional
    from dojiwick.infrastructure.ai.anthropic_client import AnthropicLLMClient
    from dojiwick.infrastructure.ai.cost_tracker import CostTracker
    from dojiwick.infrastructure.ai.llm_regime_classifier import LLMRegimeClassifier
    from dojiwick.infrastructure.ai.llm_veto_service import LLMVetoService

    client = AnthropicLLMClient(
        clock=clock,
        api_key=api_key,
        max_retries=settings.max_retries,
        timeout_sec=settings.timeout_sec,
    )
    cost_tracker = CostTracker(
        daily_budget_usd=settings.daily_budget_usd,
        clock=clock,
        input_cost_per_token=settings.input_cost_per_million / 1_000_000,
        output_cost_per_token=settings.output_cost_per_million / 1_000_000,
        metrics=metrics,
        cost_repository=cost_repository,
    )

    veto: VetoServicePort | None = None
    if settings.veto_enabled and settings.veto_model:
        veto = LLMVetoService(
            llm_client=client,
            model=settings.veto_model,
            clock=clock,
            confidence_threshold=settings.veto_confidence_threshold,
            cost_tracker=cost_tracker,
            max_response_tokens=settings.max_response_tokens,
            batch_timeout_sec=settings.batch_timeout_sec,
            metrics=metrics,
        )

    regime: AIRegimeClassifierPort | None = None
    if settings.regime_enabled and settings.regime_model:
        regime = LLMRegimeClassifier(
            llm_client=client,
            model=settings.regime_model,
            clock=clock,
            cost_tracker=cost_tracker,
            max_response_tokens=settings.max_response_tokens,
            batch_timeout_sec=settings.batch_timeout_sec,
            metrics=metrics,
        )

    log.info("AI services built veto=%s regime=%s", veto is not None, regime is not None)
    return AIServices(veto_service=veto, regime_classifier=regime, cost_tracker=cost_tracker)

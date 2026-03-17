"""LLM-backed regime classifier for ensemble detection."""

import json
import logging
from dataclasses import dataclass

import numpy as np

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.llm_client import LLMClientPort, LLMRequest
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort
from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRegimeProfile,
)
from dojiwick.infrastructure.ai.constants import MAX_RESPONSE_LENGTH, timed_llm_call
from dojiwick.infrastructure.ai.cost_tracker import CostTracker
from dojiwick.infrastructure.ai.prompts.regime_prompt import build_regime_system_prompt, build_regime_user_prompt

log = logging.getLogger(__name__)

_STATE_MAP: dict[str, int] = {
    "TRENDING_UP": MarketState.TRENDING_UP,
    "TRENDING_DOWN": MarketState.TRENDING_DOWN,
    "RANGING": MarketState.RANGING,
    "VOLATILE": MarketState.VOLATILE,
}


@dataclass(slots=True)
class LLMRegimeClassifier:
    """Implements AIRegimeClassifierPort. No cache in v1."""

    llm_client: LLMClientPort
    model: str
    clock: ClockPort
    cost_tracker: CostTracker | None = None
    max_response_tokens: int = 200
    batch_timeout_sec: float = 30.0
    metrics: MetricsSinkPort | None = None

    async def classify_batch(
        self,
        context: BatchDecisionContext,
        deterministic_regime: BatchRegimeProfile,
    ) -> BatchRegimeProfile:
        """Classify regime for each pair via LLM."""
        size = context.size
        coarse_state = np.copy(deterministic_regime.coarse_state)
        confidence = np.copy(deterministic_regime.confidence)
        valid_mask = np.copy(deterministic_regime.valid_mask)

        system_prompt = build_regime_system_prompt()
        deadline_ns = self.clock.monotonic_ns() + int(self.batch_timeout_sec * 1_000_000_000)

        for i in range(size):
            if self.clock.monotonic_ns() > deadline_ns:
                log.warning(
                    "batch timeout reached after %.1fs, echoing deterministic for remaining", self.batch_timeout_sec
                )
                break

            if not deterministic_regime.valid_mask[i]:
                continue

            user_prompt = build_regime_user_prompt(context, deterministic_regime, i)
            request = LLMRequest(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_response_tokens,
            )

            response = await timed_llm_call(self.llm_client, request, self.metrics, self.clock)

            if len(response.content) > MAX_RESPONSE_LENGTH:
                log.warning(
                    "regime response too long (%d chars), echoing deterministic pair=%s",
                    len(response.content),
                    context.market.pairs[i],
                )
                continue

            if self.cost_tracker is not None:
                await self.cost_tracker.record(
                    response.input_tokens, response.output_tokens, model=self.model, purpose="regime"
                )

            parsed_state, parsed_confidence = _parse_regime_response(response.content)
            if parsed_state is not None:
                coarse_state[i] = parsed_state
                confidence[i] = parsed_confidence

        return BatchRegimeProfile(
            coarse_state=coarse_state,
            confidence=confidence,
            valid_mask=valid_mask,
        )


def _parse_regime_response(content: str) -> tuple[int | None, float]:
    """Parse LLM JSON regime response. On failure, return None (echo deterministic)."""
    try:
        data = json.loads(content)
        state_str = str(data["state"])
        state_int = _STATE_MAP.get(state_str)
        if state_int is None:
            log.warning("unknown regime state from LLM: %s", state_str)
            return None, 0.0
        conf = float(data["confidence"])
        conf = max(0.0, min(1.0, conf))
        return state_int, conf
    except json.JSONDecodeError, KeyError, TypeError, ValueError:
        log.warning("failed to parse regime response: %s", content[:200])
        return None, 0.0

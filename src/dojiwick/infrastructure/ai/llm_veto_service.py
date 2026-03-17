"""LLM-backed veto service — default=APPROVE, LLM must justify BLOCK."""

import json
import logging
from dataclasses import dataclass

import numpy as np

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.llm_client import LLMClientPort, LLMRequest
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from dojiwick.domain.reason_codes import (
    AI_VETO_APPROVED,
    AI_VETO_BATCH_TIMEOUT,
    AI_VETO_BUDGET_EXCEEDED,
    AI_VETO_CONFIDENCE_SKIP,
    AI_VETO_CONFLICTING_REGIME,
    AI_VETO_CORRELATION_RISK,
    AI_VETO_EXTREME_VOLATILITY,
    AI_VETO_PARSE_ERROR,
    AI_VETO_STALE_SIGNAL,
)
from dojiwick.infrastructure.ai.confidence_gate import compute_llm_review_mask
from dojiwick.infrastructure.ai.constants import MAX_RESPONSE_LENGTH, timed_llm_call
from dojiwick.infrastructure.ai.cost_tracker import CostTracker
from dojiwick.infrastructure.ai.prompts.veto_prompt import build_veto_system_prompt, build_veto_user_prompt

log = logging.getLogger(__name__)

_REASON_MAP: dict[str, str] = {
    "approved": AI_VETO_APPROVED,
    "CONFLICTING_REGIME": AI_VETO_CONFLICTING_REGIME,
    "EXTREME_VOLATILITY": AI_VETO_EXTREME_VOLATILITY,
    "CORRELATION_RISK": AI_VETO_CORRELATION_RISK,
    "STALE_SIGNAL": AI_VETO_STALE_SIGNAL,
}


@dataclass(slots=True)
class LLMVetoService:
    """Implements VetoServicePort. Default=APPROVE, LLM must justify BLOCK."""

    llm_client: LLMClientPort
    model: str
    clock: ClockPort
    confidence_threshold: float = 0.85
    cost_tracker: CostTracker | None = None
    max_response_tokens: int = 200
    batch_timeout_sec: float = 30.0
    metrics: MetricsSinkPort | None = None

    async def evaluate_batch(
        self,
        context: BatchDecisionContext,
        candidates: BatchTradeCandidate,
    ) -> BatchVetoDecision:
        """Evaluate each valid candidate pair via LLM."""
        size = context.size
        approved = np.ones(size, dtype=np.bool_)
        reason_codes: list[str] = ["no_candidate"] * size

        # Determine which pairs need LLM review
        if context.regimes is not None:
            review_mask = compute_llm_review_mask(candidates, context.regimes, self.confidence_threshold)
        else:
            review_mask = candidates.valid_mask.copy()

        system_prompt = build_veto_system_prompt()
        deadline_ns = self.clock.monotonic_ns() + int(self.batch_timeout_sec * 1_000_000_000)

        for i in range(size):
            if self.clock.monotonic_ns() > deadline_ns:
                log.warning("batch timeout reached after %.1fs, auto-approving remaining", self.batch_timeout_sec)
                for j in range(i, size):
                    if candidates.valid_mask[j] and review_mask[j]:
                        reason_codes[j] = AI_VETO_BATCH_TIMEOUT
                break

            if not candidates.valid_mask[i]:
                reason_codes[i] = "no_candidate"
                continue

            if not review_mask[i]:
                reason_codes[i] = AI_VETO_CONFIDENCE_SKIP
                continue

            if self.cost_tracker is not None and self.cost_tracker.is_budget_exceeded():
                reason_codes[i] = AI_VETO_BUDGET_EXCEEDED
                log.warning("daily budget exceeded, auto-approving pair=%s", context.market.pairs[i])
                continue

            user_prompt = build_veto_user_prompt(context, candidates, i, regimes=context.regimes)
            request = LLMRequest(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_response_tokens,
            )

            response = await timed_llm_call(self.llm_client, request, self.metrics, self.clock)

            if len(response.content) > MAX_RESPONSE_LENGTH:
                log.warning(
                    "veto response too long (%d chars), auto-approving pair=%s",
                    len(response.content),
                    context.market.pairs[i],
                )
                reason_codes[i] = AI_VETO_PARSE_ERROR
                continue

            if self.cost_tracker is not None:
                await self.cost_tracker.record(
                    response.input_tokens, response.output_tokens, model=self.model, purpose="veto"
                )

            is_approved, reason = _parse_veto_response(response.content)
            approved[i] = is_approved
            reason_codes[i] = reason

        return BatchVetoDecision(approved_mask=approved, reason_codes=tuple(reason_codes))


def _parse_veto_response(content: str) -> tuple[bool, str]:
    """Parse LLM JSON response. On failure, approve (fail-open)."""
    try:
        data = json.loads(content)
        is_approved = bool(data["approved"])
        raw_reason = str(data["reason"])
        reason = _REASON_MAP.get(raw_reason, AI_VETO_PARSE_ERROR)
        if reason == AI_VETO_PARSE_ERROR:
            log.warning("unknown veto reason from LLM: %s, approving", raw_reason)
            return True, AI_VETO_PARSE_ERROR
        return is_approved, reason
    except json.JSONDecodeError, KeyError, TypeError:
        log.warning("failed to parse veto response: %s, approving", content[:200])
        return True, AI_VETO_PARSE_ERROR

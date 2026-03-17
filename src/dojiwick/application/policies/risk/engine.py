"""Risk policy engine -- evaluates all rules, merges by precedence."""

import numpy as np

from dojiwick.compute.kernels.risk.rule import RiskRule
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchRiskAssessment,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams
from dojiwick.domain.reason_codes import RISK_NO_CANDIDATE, RISK_OK


class RiskPolicyEngine:
    """Composable risk engine that evaluates pluggable rules and merges results.

    Rules are registered via :meth:`register` and evaluated in
    :meth:`assess_risk`. Each rule produces a :class:`RiskRuleDecision`
    with a ``blocked_mask``, ``precedence``, ``risk_score``, and
    ``reason_code``.  The engine merges decisions so that for every row
    the highest-priority (lowest ``precedence`` value) blocking rule wins.

    ``RISK_NO_CANDIDATE`` is handled as a hard-coded sentinel before any
    pluggable rules execute (precedence 0, score 1.0).
    """

    def __init__(self) -> None:
        self._rules: list[RiskRule] = []

    def register(self, rule: RiskRule) -> None:
        """Add a rule to the engine."""
        self._rules.append(rule)

    def assess_risk(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> BatchRiskAssessment:
        """Evaluate all registered rules and merge into a single assessment."""
        size = context.size

        # Seed defaults: every row starts as allowed / OK
        reason: list[str] = [RISK_OK] * size
        score = np.zeros(size, dtype=np.float64)

        # --- Hard-coded gate: no-candidate rows (precedence 0, score 1.0) ---
        no_candidate_mask = ~candidate.valid_mask

        # --- Collect pluggable rule decisions ---
        decisions: list[RiskRuleDecision] = []
        for rule in self._rules:
            decision = rule.evaluate(
                context=context,
                candidate=candidate,
                risk_params=risk_params,
            )
            decisions.append(decision)

        # Sort by precedence ascending (lower number = higher priority)
        decisions.sort(key=lambda d: d.precedence)

        # --- Merge: apply in reverse precedence order so highest priority wins ---
        for decision in reversed(decisions):
            effective_block = decision.blocked_mask & candidate.valid_mask
            for idx in np.flatnonzero(effective_block):
                reason[idx] = decision.reason_code
            score[effective_block] = decision.risk_score

        # Apply no-candidate last (highest priority, precedence 0)
        for idx in np.flatnonzero(no_candidate_mask):
            reason[idx] = RISK_NO_CANDIDATE
        score[no_candidate_mask] = 1.0

        # --- Build allowed mask: rows that are still RISK_OK ---
        allowed = np.array([r == RISK_OK for r in reason], dtype=np.bool_)

        return BatchRiskAssessment(
            allowed_mask=allowed,
            reason_codes=tuple(reason),
            risk_score=score,
        )

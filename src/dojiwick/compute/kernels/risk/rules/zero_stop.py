"""Zero stop distance risk rule — blocks candidates with no stop distance."""

import numpy as np

from dojiwick.compute.kernels.risk.rule import ConfigurableRiskRule
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams
from dojiwick.domain.reason_codes import RISK_ZERO_STOP_DISTANCE


class ZeroStopRule(ConfigurableRiskRule):
    """Checks for zero stop distance on valid candidates.

    Computes the absolute distance between entry and stop price. Rows with
    a valid candidate mask and zero stop distance are blocked.
    """

    @property
    def name(self) -> str:
        return "zero_stop"

    def evaluate(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> RiskRuleDecision:
        stop_distance = np.abs(candidate.entry_price - candidate.stop_price)
        blocked = candidate.valid_mask & (stop_distance == 0.0)

        return RiskRuleDecision(
            rule_name=self.name,
            blocked_mask=blocked,
            reason_code=RISK_ZERO_STOP_DISTANCE,
            precedence=self._precedence,
            risk_score=self._risk_score,
        )

"""Max positions risk rule — blocks when open position count reaches limit."""

import numpy as np

from dojiwick.compute.kernels.risk.rule import ConfigurableRiskRule
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams
from dojiwick.domain.reason_codes import RISK_MAX_POSITIONS


class MaxPositionsRule(ConfigurableRiskRule):
    """Checks if the number of open positions has reached the configured maximum.

    Compares ``portfolio.open_positions_total`` against
    the per-pair ``max_open_positions``. Rows where the limit is reached are blocked.
    """

    @property
    def name(self) -> str:
        return "max_positions"

    def evaluate(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> RiskRuleDecision:
        threshold = np.array([rp.max_open_positions for rp in risk_params])
        blocked = context.portfolio.open_positions_total >= threshold

        return RiskRuleDecision(
            rule_name=self.name,
            blocked_mask=blocked,
            reason_code=RISK_MAX_POSITIONS,
            precedence=self._precedence,
            risk_score=self._risk_score,
        )

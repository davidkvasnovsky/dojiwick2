"""Daily loss risk rule — blocks trading when daily PnL loss exceeds threshold."""

import numpy as np

from dojiwick.compute.kernels.risk.rule import ConfigurableRiskRule
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams
from dojiwick.domain.reason_codes import RISK_DAILY_LOSS


class DailyLossRule(ConfigurableRiskRule):
    """Checks if daily PnL loss exceeds the configured threshold.

    Computes the percentage change from day-start equity to current equity.
    If the loss exceeds the per-pair ``max_daily_loss_pct``, the row is blocked.
    """

    @property
    def name(self) -> str:
        return "daily_loss"

    def evaluate(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> RiskRuleDecision:
        day_start = context.portfolio.day_start_equity_usd
        # A blown account (day_start 0) must block, not divide to nan —
        # nan <= threshold is False and would silently disable this gate.
        alive = day_start > 0.0
        daily_pnl_pct = np.where(
            alive,
            (context.portfolio.equity_usd / np.where(alive, day_start, 1.0) - 1.0) * 100.0,
            -100.0,
        )
        threshold = np.array([rp.max_daily_loss_pct for rp in risk_params])
        blocked = daily_pnl_pct <= -threshold

        return RiskRuleDecision(
            rule_name=self.name,
            blocked_mask=blocked,
            reason_code=RISK_DAILY_LOSS,
            precedence=self._precedence,
            risk_score=self._risk_score,
        )

"""Minimum risk-reward ratio rule — blocks candidates below the threshold."""

import numpy as np

from dojiwick.compute.kernels.risk.rule import ConfigurableRiskRule
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams
from dojiwick.domain.reason_codes import RISK_MIN_RR


class MinRRRule(ConfigurableRiskRule):
    """Checks that the risk-reward ratio meets the configured minimum.

    Computes direction-aware risk-reward for BUY and SHORT rows. Rows
    with a valid candidate, non-zero stop distance, and RR below
    the per-pair ``min_rr_ratio`` are blocked.
    """

    @property
    def name(self) -> str:
        return "min_rr"

    def evaluate(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> RiskRuleDecision:
        size = context.size
        rr = np.zeros(size, dtype=np.float64)

        buy_rows = candidate.action == TradeAction.BUY.value
        short_rows = candidate.action == TradeAction.SHORT.value

        # Stop distance guard — exclude zero-stop rows from RR computation
        stop_distance = np.abs(candidate.entry_price - candidate.stop_price)
        zero_stop = candidate.valid_mask & (stop_distance == 0.0)

        # BUY: reward = TP - entry, risk = entry - stop
        buy_den = candidate.entry_price - candidate.stop_price
        buy_num = candidate.take_profit_price - candidate.entry_price
        rr[buy_rows] = np.divide(
            buy_num[buy_rows],
            buy_den[buy_rows],
            out=np.zeros(np.count_nonzero(buy_rows), dtype=np.float64),
            where=buy_den[buy_rows] > 0,
        )

        # SHORT: reward = entry - TP, risk = stop - entry
        short_den = candidate.stop_price - candidate.entry_price
        short_num = candidate.entry_price - candidate.take_profit_price
        rr[short_rows] = np.divide(
            short_num[short_rows],
            short_den[short_rows],
            out=np.zeros(np.count_nonzero(short_rows), dtype=np.float64),
            where=short_den[short_rows] > 0,
        )

        # Block rows with valid candidate, non-zero stop, and RR below threshold
        threshold = np.array([rp.min_rr_ratio for rp in risk_params])
        blocked = candidate.valid_mask & ~zero_stop & (rr < threshold)

        return RiskRuleDecision(
            rule_name=self.name,
            blocked_mask=blocked,
            reason_code=RISK_MIN_RR,
            precedence=self._precedence,
            risk_score=self._risk_score,
        )

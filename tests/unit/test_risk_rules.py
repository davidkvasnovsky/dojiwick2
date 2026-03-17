"""Unit tests for the 4 built-in risk rules and per-pair risk engine."""

import numpy as np

from dojiwick.application.policies.risk.engine import RiskPolicyEngine
from dojiwick.compute.kernels.risk.rules.daily_loss import DailyLossRule
from dojiwick.compute.kernels.risk.rules.max_positions import MaxPositionsRule
from dojiwick.compute.kernels.risk.rules.min_rr import MinRRRule
from dojiwick.compute.kernels.risk.rules.zero_stop import ZeroStopRule
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.models.value_objects.batch_models import BatchTradeCandidate
from fixtures.factories.infrastructure import default_risk_params
from dojiwick.domain.reason_codes import (
    RISK_DAILY_LOSS,
    RISK_MAX_POSITIONS,
    RISK_MIN_RR,
    RISK_OK,
    RISK_ZERO_STOP_DISTANCE,
)

from fixtures.factories.domain import ContextBuilder


def _candidate(
    action: int = TradeAction.BUY.value,
    entry: float = 100.0,
    stop: float = 95.0,
    tp: float = 110.0,
) -> BatchTradeCandidate:
    return BatchTradeCandidate(
        action=np.array([action], dtype=np.int64),
        entry_price=np.array([entry], dtype=np.float64),
        stop_price=np.array([stop], dtype=np.float64),
        take_profit_price=np.array([tp], dtype=np.float64),
        strategy_name=("trend_follow",),
        strategy_variant=("baseline",),
        reason_codes=("signal",),
        valid_mask=np.array([True], dtype=np.bool_),
    )


_PARAMS = default_risk_params()


class TestZeroStopRule:
    rule = ZeroStopRule(precedence=3, risk_score=0.9)

    def test_blocks_zero_distance(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        cand = _candidate(entry=100.0, stop=100.0)
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert result.blocked_mask[0]
        assert result.reason_code == RISK_ZERO_STOP_DISTANCE

    def test_allows_nonzero_distance(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        cand = _candidate(entry=100.0, stop=95.0)
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert not result.blocked_mask[0]


class TestMinRRRule:
    rule = MinRRRule(precedence=4, risk_score=0.7)

    def test_blocks_low_ratio_buy(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        # RR = (101 - 100) / (100 - 95) = 0.2, well below min_rr_ratio=1.3
        cand = _candidate(action=TradeAction.BUY.value, entry=100.0, stop=95.0, tp=101.0)
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert result.blocked_mask[0]
        assert result.reason_code == RISK_MIN_RR

    def test_allows_good_ratio_buy(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
        # RR = (110 - 100) / (100 - 95) = 2.0, above min_rr_ratio=1.3
        cand = _candidate(action=TradeAction.BUY.value, entry=100.0, stop=95.0, tp=110.0)
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert not result.blocked_mask[0]


class TestMaxPositionsRule:
    rule = MaxPositionsRule(precedence=2, risk_score=0.8)

    def test_blocks_at_limit(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).with_open_positions_total([8]).build()
        cand = _candidate()
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert result.blocked_mask[0]
        assert result.reason_code == RISK_MAX_POSITIONS

    def test_allows_below_limit(self) -> None:
        ctx = ContextBuilder(pairs=("BTC/USDC",)).with_open_positions_total([3]).build()
        cand = _candidate()
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert not result.blocked_mask[0]


class TestDailyLossRule:
    rule = DailyLossRule(precedence=1, risk_score=1.0)

    def test_blocks_exceeded(self) -> None:
        # equity=900, day_start=1000 → -10% loss, exceeds max_daily_loss_pct=5%
        ctx = ContextBuilder(pairs=("BTC/USDC",)).with_equity([900.0]).with_day_start_equity([1000.0]).build()
        cand = _candidate()
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert result.blocked_mask[0]
        assert result.reason_code == RISK_DAILY_LOSS

    def test_allows_within_threshold(self) -> None:
        # equity=980, day_start=1000 → -2% loss, within max_daily_loss_pct=5%
        ctx = ContextBuilder(pairs=("BTC/USDC",)).with_equity([980.0]).with_day_start_equity([1000.0]).build()
        cand = _candidate()
        result = self.rule.evaluate(context=ctx, candidate=cand, risk_params=(_PARAMS,))
        assert not result.blocked_mask[0]


class TestPerPairRiskEngine:
    """Test that per_pair_risk applies different thresholds per pair."""

    def test_per_pair_max_positions_different_limits(self) -> None:
        """Row 0 has limit=2, row 1 has limit=10. With 3 open positions, row 0 blocked, row 1 allowed."""
        ctx = ContextBuilder(pairs=("BTC/USDC", "ETH/USDC")).with_open_positions_total([3, 3]).build()
        cand = BatchTradeCandidate(
            action=np.array([TradeAction.BUY.value, TradeAction.BUY.value], dtype=np.int64),
            entry_price=np.array([100.0, 100.0], dtype=np.float64),
            stop_price=np.array([95.0, 95.0], dtype=np.float64),
            take_profit_price=np.array([110.0, 110.0], dtype=np.float64),
            strategy_name=("trend_follow", "trend_follow"),
            strategy_variant=("baseline", "baseline"),
            reason_codes=("signal", "signal"),
            valid_mask=np.array([True, True], dtype=np.bool_),
        )

        strict_risk = default_risk_params(max_open_positions=2)
        lax_risk = default_risk_params(max_open_positions=10)

        engine = RiskPolicyEngine()
        engine.register(MaxPositionsRule(precedence=2, risk_score=0.8))

        result = engine.assess_risk(
            context=ctx,
            candidate=cand,
            risk_params=(strict_risk, lax_risk),
        )

        # Row 0 blocked (3 >= 2), row 1 allowed (3 < 10)
        assert result.reason_codes[0] == RISK_MAX_POSITIONS
        assert result.reason_codes[1] == RISK_OK

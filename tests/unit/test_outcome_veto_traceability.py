"""Tests for outcome veto traceability — specific reason codes, not generic AI_VETO."""

import numpy as np

from dojiwick.domain.enums import DecisionAuthority, DecisionStatus, ExecutionStatus
from dojiwick.domain.models.value_objects.batch_models import (
    BatchExecutionIntent,
    BatchRegimeProfile,
    BatchRiskAssessment,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.reason_codes import AI_VETO_CONFLICTING_REGIME, AI_VETO_EXTREME_VOLATILITY
from dojiwick.application.orchestration.outcome_assembler import OutcomeInputs, build_outcomes
from fixtures.factories.domain import ContextBuilder


def _make_veto_inputs(
    veto_reasons: tuple[str, ...],
    approved: list[bool],
) -> OutcomeInputs:
    size = len(approved)
    ctx = ContextBuilder(pairs=tuple(f"PAIR{i}/USDC" for i in range(size))).build()
    target_ids = tuple(f"target_{i}" for i in range(size))
    return OutcomeInputs(
        context=ctx,
        regime=BatchRegimeProfile(
            coarse_state=np.ones(size, dtype=np.int64),
            confidence=np.full(size, 0.9, dtype=np.float64),
            valid_mask=np.ones(size, dtype=np.bool_),
        ),
        candidate=BatchTradeCandidate(
            action=np.ones(size, dtype=np.int64),
            entry_price=np.full(size, 100.0, dtype=np.float64),
            stop_price=np.full(size, 95.0, dtype=np.float64),
            take_profit_price=np.full(size, 110.0, dtype=np.float64),
            strategy_name=tuple("trend" for _ in range(size)),
            strategy_variant=tuple("baseline" for _ in range(size)),
            reason_codes=tuple("signal" for _ in range(size)),
            valid_mask=np.ones(size, dtype=np.bool_),
        ),
        veto=BatchVetoDecision(
            approved_mask=np.array(approved, dtype=np.bool_),
            reason_codes=veto_reasons,
        ),
        risk=BatchRiskAssessment(
            allowed_mask=np.ones(size, dtype=np.bool_),
            reason_codes=tuple("risk_ok" for _ in range(size)),
            risk_score=np.zeros(size, dtype=np.float64),
        ),
        intent=BatchExecutionIntent(
            pairs=ctx.market.pairs,
            action=np.ones(size, dtype=np.int64),
            quantity=np.full(size, 0.1, dtype=np.float64),
            notional_usd=np.full(size, 100.0, dtype=np.float64),
            entry_price=np.full(size, 100.0, dtype=np.float64),
            stop_price=np.full(size, 95.0, dtype=np.float64),
            take_profit_price=np.full(size, 110.0, dtype=np.float64),
            strategy_name=tuple("trend" for _ in range(size)),
            strategy_variant=tuple("baseline" for _ in range(size)),
            active_mask=np.ones(size, dtype=np.bool_),
        ),
        receipts=tuple(ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="vetoed") for _ in range(size)),
        authority=DecisionAuthority.DETERMINISTIC_PLUS_AI_VETO,
        config_hash="abc123",
        target_ids=target_ids,
    )


class TestOutcomeVetoTraceability:
    def test_vetoed_outcome_carries_specific_reason(self) -> None:
        inputs = _make_veto_inputs(
            veto_reasons=(AI_VETO_CONFLICTING_REGIME,),
            approved=[False],
        )
        outcomes = build_outcomes(inputs)
        assert outcomes[0].status == DecisionStatus.VETOED
        assert outcomes[0].reason_code == AI_VETO_CONFLICTING_REGIME

    def test_vetoed_with_extreme_volatility(self) -> None:
        inputs = _make_veto_inputs(
            veto_reasons=(AI_VETO_EXTREME_VOLATILITY,),
            approved=[False],
        )
        outcomes = build_outcomes(inputs)
        assert outcomes[0].reason_code == AI_VETO_EXTREME_VOLATILITY

    def test_multi_pair_different_reasons(self) -> None:
        inputs = _make_veto_inputs(
            veto_reasons=(AI_VETO_CONFLICTING_REGIME, AI_VETO_EXTREME_VOLATILITY),
            approved=[False, False],
        )
        outcomes = build_outcomes(inputs)
        assert outcomes[0].reason_code == AI_VETO_CONFLICTING_REGIME
        assert outcomes[1].reason_code == AI_VETO_EXTREME_VOLATILITY

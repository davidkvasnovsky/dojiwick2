"""Outcome factory unit tests."""

from datetime import UTC, datetime

import numpy as np
import pytest

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
    BatchRegimeProfile,
    BatchRiskAssessment,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from decimal import Decimal

from dojiwick.domain.enums import DecisionAuthority, DecisionStatus, ExecutionStatus, MarketState, TradeAction
from dojiwick.domain.errors import DomainValidationError
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.application.orchestration.outcome_assembler import OutcomeInputs, build_outcomes


def _make_inputs(
    *,
    size: int = 2,
    actions: list[int] | None = None,
    valid_mask: list[bool] | None = None,
    approved_mask: list[bool] | None = None,
    allowed_mask: list[bool] | None = None,
    active_mask: list[bool] | None = None,
    receipts: tuple[ExecutionReceipt, ...] | None = None,
) -> OutcomeInputs:
    pairs = ("BTC/USDC", "ETH/USDC")[:size]
    now = datetime.now(UTC)

    _actions = actions or [TradeAction.HOLD.value] * size
    _valid = valid_mask if valid_mask is not None else [False] * size
    _approved = approved_mask if approved_mask is not None else [True] * size
    _allowed = allowed_mask if allowed_mask is not None else [True] * size
    _active = active_mask if active_mask is not None else [False] * size

    from dojiwick.domain.indicator_schema import INDICATOR_COUNT

    indicators = np.full((size, INDICATOR_COUNT), 50.0, dtype=np.float64)
    context = BatchDecisionContext(
        market=BatchMarketSnapshot(
            pairs=pairs,
            observed_at=now,
            price=np.full(size, 100.0, dtype=np.float64),
            indicators=indicators,
        ),
        portfolio=BatchPortfolioSnapshot(
            equity_usd=np.full(size, 1000.0, dtype=np.float64),
            day_start_equity_usd=np.full(size, 1000.0, dtype=np.float64),
            open_positions_total=np.zeros(size, dtype=np.int64),
            has_open_position=np.zeros(size, dtype=np.bool_),
            unrealized_pnl_usd=np.zeros(size, dtype=np.float64),
        ),
    )

    regime = BatchRegimeProfile(
        coarse_state=np.full(size, MarketState.RANGING.value, dtype=np.int64),
        confidence=np.full(size, 0.8, dtype=np.float64),
        valid_mask=np.ones(size, dtype=np.bool_),
    )

    candidate = BatchTradeCandidate(
        action=np.array(_actions, dtype=np.int64),
        entry_price=np.full(size, 100.0, dtype=np.float64),
        stop_price=np.full(size, 99.0, dtype=np.float64),
        take_profit_price=np.full(size, 102.0, dtype=np.float64),
        strategy_name=tuple("trend_follow" for _ in range(size)),
        strategy_variant=tuple("baseline" for _ in range(size)),
        reason_codes=tuple("strategy_signal" for _ in range(size)),
        valid_mask=np.array(_valid, dtype=np.bool_),
    )

    veto = BatchVetoDecision(
        approved_mask=np.array(_approved, dtype=np.bool_),
        reason_codes=tuple("veto_approved" for _ in range(size)),
    )

    risk = BatchRiskAssessment(
        allowed_mask=np.array(_allowed, dtype=np.bool_),
        reason_codes=tuple("risk_ok" for _ in range(size)),
        risk_score=np.full(size, 1.0, dtype=np.float64),
    )

    intent = BatchExecutionIntent(
        pairs=pairs,
        action=np.array(_actions, dtype=np.int64),
        quantity=np.full(size, 0.1, dtype=np.float64),
        notional_usd=np.full(size, 10.0, dtype=np.float64),
        entry_price=np.full(size, 100.0, dtype=np.float64),
        stop_price=np.full(size, 99.0, dtype=np.float64),
        take_profit_price=np.full(size, 102.0, dtype=np.float64),
        strategy_name=tuple("trend_follow" for _ in range(size)),
        strategy_variant=tuple("baseline" for _ in range(size)),
        active_mask=np.array(_active, dtype=np.bool_),
    )

    if receipts is None:
        receipts = tuple(
            ExecutionReceipt(
                status=ExecutionStatus.FILLED if _active[i] else ExecutionStatus.SKIPPED,
                reason="filled" if _active[i] else "inactive",
                fill_price=Decimal("100") if _active[i] else None,
                filled_quantity=Decimal("0.1") if _active[i] else Decimal(0),
                exchange_timestamp=now if _active[i] else None,
            )
            for i in range(size)
        )

    target_ids = tuple(f"target_{i}" for i in range(size))

    return OutcomeInputs(
        context=context,
        regime=regime,
        candidate=candidate,
        veto=veto,
        risk=risk,
        intent=intent,
        receipts=receipts,
        authority=DecisionAuthority.DETERMINISTIC_ONLY,
        config_hash="test-config-hash",
        target_ids=target_ids,
    )


def test_all_hold_batch() -> None:
    inputs = _make_inputs(size=2)
    outcomes = build_outcomes(inputs)

    assert len(outcomes) == 2
    assert all(o.status == DecisionStatus.HOLD for o in outcomes)


def test_mixed_vetoed_and_executed() -> None:
    inputs = _make_inputs(
        size=2,
        actions=[TradeAction.BUY.value, TradeAction.BUY.value],
        valid_mask=[True, True],
        approved_mask=[False, True],
        allowed_mask=[True, True],
        active_mask=[False, True],
    )
    outcomes = build_outcomes(inputs)

    assert outcomes[0].status == DecisionStatus.VETOED
    assert outcomes[1].status == DecisionStatus.EXECUTED


def test_risk_blocked() -> None:
    inputs = _make_inputs(
        size=1,
        actions=[TradeAction.BUY.value],
        valid_mask=[True],
        approved_mask=[True],
        allowed_mask=[False],
        active_mask=[False],
    )
    outcomes = build_outcomes(inputs)

    assert outcomes[0].status == DecisionStatus.BLOCKED_RISK


def test_receipt_alignment_mismatch_raises() -> None:
    inputs = _make_inputs(size=2)
    bad_inputs = OutcomeInputs(
        context=inputs.context,
        regime=inputs.regime,
        candidate=inputs.candidate,
        veto=inputs.veto,
        risk=inputs.risk,
        intent=inputs.intent,
        receipts=(ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="x"),),
        authority=inputs.authority,
        config_hash=inputs.config_hash,
        target_ids=inputs.target_ids,
    )

    with pytest.raises(ValueError, match="receipts must align"):
        build_outcomes(bad_inputs)


def test_invalid_market_state_code_raises() -> None:
    inputs = _make_inputs(size=1)
    bad_regime = BatchRegimeProfile(
        coarse_state=np.array([999], dtype=np.int64),
        confidence=np.array([0.8], dtype=np.float64),
        valid_mask=np.array([True], dtype=np.bool_),
    )
    bad_inputs = OutcomeInputs(
        context=inputs.context,
        regime=bad_regime,
        candidate=inputs.candidate,
        veto=inputs.veto,
        risk=inputs.risk,
        intent=inputs.intent,
        receipts=inputs.receipts,
        authority=inputs.authority,
        config_hash=inputs.config_hash,
        target_ids=inputs.target_ids,
    )

    with pytest.raises(DomainValidationError, match="invalid market state"):
        build_outcomes(bad_inputs)

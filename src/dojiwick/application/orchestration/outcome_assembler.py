"""Transforms batch compute artifacts into persisted outcomes."""

from dataclasses import dataclass

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchRegimeProfile,
    BatchRiskAssessment,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from dojiwick.domain.enums import (
    DecisionAuthority,
    DecisionStatus,
    ExecutionStatus,
    MarketState,
    TradeAction,
)
from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome, ExecutionReceipt
from dojiwick.domain.errors import DomainValidationError
from dojiwick.domain.numerics import to_money, to_price, to_quantity
from dojiwick.domain.reason_codes import (
    EXECUTION_ERROR,
    EXECUTION_FILLED,
    EXECUTION_REJECTED,
    EXECUTION_SKIPPED,
    RISK_NO_CANDIDATE,
    STRATEGY_HOLD,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class OutcomeInputs:
    """Grouped inputs for outcome synthesis."""

    context: BatchDecisionContext
    regime: BatchRegimeProfile
    candidate: BatchTradeCandidate
    veto: BatchVetoDecision
    risk: BatchRiskAssessment
    intent: BatchExecutionIntent
    receipts: tuple[ExecutionReceipt, ...]
    authority: DecisionAuthority
    config_hash: str
    target_ids: tuple[str, ...]
    tick_id: str = ""
    confidence_raw: tuple[float, ...] = ()


def build_outcomes(inputs: OutcomeInputs) -> tuple[DecisionOutcome, ...]:
    """Build per-pair outcomes in stable batch order."""

    size = inputs.context.size
    if len(inputs.receipts) != size:
        raise ValueError("receipts must align with batch size")
    if len(inputs.target_ids) != size:
        raise ValueError(f"target_ids length ({len(inputs.target_ids)}) must match batch size ({size})")

    return tuple(_build_one(inputs, index) for index in range(size))


def _build_one(inputs: OutcomeInputs, index: int) -> DecisionOutcome:
    """Build a single outcome from batch artifacts at the given index."""

    status, reason, note = _resolve_decision(inputs, index)

    return DecisionOutcome(
        pair=inputs.context.market.pairs[index],
        target_id=inputs.target_ids[index],
        observed_at=inputs.context.market.observed_at,
        authority=inputs.authority,
        status=status,
        market_state=_market_state_from_code(int(inputs.regime.coarse_state[index])),
        action=_trade_action_from_code(int(inputs.candidate.action[index])),
        strategy_name=inputs.intent.strategy_name[index],
        strategy_variant=inputs.intent.strategy_variant[index],
        reason_code=reason,
        confidence=float(inputs.regime.confidence[index]),
        entry_price=to_price(float(inputs.intent.entry_price[index])),
        stop_price=to_price(float(inputs.intent.stop_price[index])),
        take_profit_price=to_price(float(inputs.intent.take_profit_price[index])),
        quantity=to_quantity(float(inputs.intent.quantity[index])),
        notional_usd=to_money(float(inputs.intent.notional_usd[index])),
        config_hash=inputs.config_hash,
        note=note,
        tick_id=inputs.tick_id,
        confidence_raw=inputs.confidence_raw[index] if inputs.confidence_raw else 0.0,
    )


def _resolve_decision(inputs: OutcomeInputs, index: int) -> tuple[DecisionStatus, str, str]:
    """Determine decision status via priority cascade with early returns."""

    if not inputs.candidate.valid_mask[index]:
        return DecisionStatus.HOLD, STRATEGY_HOLD, ""

    if not inputs.veto.approved_mask[index]:
        return DecisionStatus.VETOED, inputs.veto.reason_codes[index], ""

    if not inputs.risk.allowed_mask[index]:
        reason = inputs.risk.reason_codes[index]
        status = DecisionStatus.HOLD if reason == RISK_NO_CANDIDATE else DecisionStatus.BLOCKED_RISK
        return status, reason, ""

    if not inputs.intent.active_mask[index]:
        return DecisionStatus.HOLD, inputs.candidate.reason_codes[index], ""

    return _resolve_execution(inputs.receipts[index])


def _resolve_execution(receipt: ExecutionReceipt) -> tuple[DecisionStatus, str, str]:
    """Map execution receipt status to decision outcome."""

    match receipt.status:
        case ExecutionStatus.FILLED:
            return DecisionStatus.EXECUTED, EXECUTION_FILLED, ""
        case ExecutionStatus.REJECTED:
            return DecisionStatus.BLOCKED_RISK, EXECUTION_REJECTED, receipt.reason
        case ExecutionStatus.SKIPPED:
            return DecisionStatus.HOLD, EXECUTION_SKIPPED, receipt.reason
        case ExecutionStatus.CANCELLED:
            return DecisionStatus.HOLD, EXECUTION_SKIPPED, receipt.reason
        case ExecutionStatus.ERROR:
            return DecisionStatus.ERROR, EXECUTION_ERROR, receipt.reason


def _market_state_from_code(value: int) -> MarketState:
    try:
        return MarketState(value)
    except ValueError as exc:
        raise DomainValidationError(f"invalid market state code: {value}") from exc


def _trade_action_from_code(value: int) -> TradeAction:
    try:
        return TradeAction(value)
    except ValueError as exc:
        raise DomainValidationError(f"invalid trade action code: {value}") from exc

"""Unit tests for planner receipt alignment in TickService."""

from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.application.orchestration.target_resolver import ResolvedTargets
from dojiwick.application.use_cases.run_tick import align_plan_receipts
from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderSide,
    OrderType,
    PositionSide,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, TargetLegPosition
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from dojiwick.domain.numerics import to_price, to_quantity


def _instrument() -> InstrumentId:
    return InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="BTCUSDC",
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )


def _filled(reason: str) -> ExecutionReceipt:
    return ExecutionReceipt(
        status=ExecutionStatus.FILLED,
        reason=reason,
        fill_price=to_price("1"),
        filled_quantity=to_quantity("1"),
        exchange_timestamp=datetime.now(UTC),
    )


def test_align_plan_receipts_aggregates_multi_delta_target() -> None:
    instrument = _instrument()
    resolved = ResolvedTargets(
        targets=(
            TargetLegPosition(
                account="default",
                instrument_id=instrument,
                position_side=PositionSide.NET,
                target_qty=Decimal("-0.1"),
            ),
        ),
        batch_indices=(0,),
    )
    plan = ExecutionPlan(
        account="default",
        deltas=(
            LegDelta(
                instrument_id=instrument,
                target_index=0,
                position_side=PositionSide.NET,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.3"),
            ),
            LegDelta(
                instrument_id=instrument,
                target_index=0,
                position_side=PositionSide.NET,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            ),
        ),
    )

    aligned = align_plan_receipts(
        batch_size=1,
        resolved=resolved,
        plan=plan,
        plan_receipts=(_filled("close"), _filled("open")),
    )

    assert len(aligned) == 1
    assert aligned[0].status is ExecutionStatus.FILLED
    assert aligned[0].reason == "open"


def test_align_plan_receipts_marks_missing_delta_receipt_as_error() -> None:
    instrument = _instrument()
    resolved = ResolvedTargets(
        targets=(
            TargetLegPosition(
                account="default",
                instrument_id=instrument,
                position_side=PositionSide.NET,
                target_qty=Decimal("0.1"),
            ),
        ),
        batch_indices=(0,),
    )
    plan = ExecutionPlan(
        account="default",
        deltas=(
            LegDelta(
                instrument_id=instrument,
                target_index=0,
                position_side=PositionSide.NET,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            ),
        ),
    )

    aligned = align_plan_receipts(
        batch_size=1,
        resolved=resolved,
        plan=plan,
        plan_receipts=(),
    )

    assert len(aligned) == 1
    assert aligned[0].status is ExecutionStatus.ERROR
    assert aligned[0].reason == "missing_plan_receipt"

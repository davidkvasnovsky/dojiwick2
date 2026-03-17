"""Unit tests for OrderLedgerService."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from dojiwick.application.services.order_ledger import OrderLedgerService
from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderEventType,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.fill_repository import FakeFillRepo
from fixtures.fakes.instrument_repository import FakeInstrumentRepo
from fixtures.fakes.order_event_repository import FakeOrderEventRepository
from fixtures.fakes.order_report_repository import FakeOrderReportRepo
from fixtures.fakes.order_request_repository import FakeOrderRequestRepo

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_IID = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDC",
    base_asset="BTC",
    quote_asset="USDC",
    settle_asset="USDC",
)


def _make_service() -> tuple[
    OrderLedgerService,
    FakeInstrumentRepo,
    FakeOrderRequestRepo,
    FakeOrderReportRepo,
    FakeFillRepo,
    FakeOrderEventRepository,
]:
    instrument_repo = FakeInstrumentRepo()
    instrument_repo.seed(BINANCE_VENUE, BINANCE_USD_C, "BTCUSDC", db_id=42)
    request_repo = FakeOrderRequestRepo()
    report_repo = FakeOrderReportRepo()
    fill_repo = FakeFillRepo()
    event_repo = FakeOrderEventRepository()
    svc = OrderLedgerService(
        instrument_repo=instrument_repo,
        order_request_repo=request_repo,
        order_report_repo=report_repo,
        fill_repo=fill_repo,
        order_event_repo=event_repo,
        clock=FixedClock(_NOW),
    )
    return svc, instrument_repo, request_repo, report_repo, fill_repo, event_repo


def _plan(*deltas: LegDelta) -> ExecutionPlan:
    return ExecutionPlan(account="default", deltas=deltas)


def _delta(
    *,
    side: OrderSide = OrderSide.BUY,
    qty: Decimal = Decimal("0.1"),
    price: Decimal | None = Decimal("50000"),
    order_type: OrderType = OrderType.MARKET,
    position_side: PositionSide = PositionSide.LONG,
    reduce_only: bool = False,
    close_position: bool = False,
    sequence: int = 0,
    target_index: int = 0,
) -> LegDelta:
    return LegDelta(
        instrument_id=_IID,
        target_index=target_index,
        position_side=position_side,
        side=side,
        order_type=order_type,
        quantity=qty,
        price=price,
        reduce_only=reduce_only,
        close_position=close_position,
        sequence=sequence,
    )


def _filled_receipt(
    price: Decimal = Decimal("50000"),
    qty: Decimal = Decimal("0.1"),
    order_id: str = "exch_123",
) -> ExecutionReceipt:
    return ExecutionReceipt(
        status=ExecutionStatus.FILLED,
        reason="filled",
        fill_price=price,
        filled_quantity=qty,
        order_id=order_id,
        exchange_timestamp=_NOW,
    )


@pytest.mark.asyncio
async def test_filled_receipt_records_request_report_fill_event() -> None:
    svc, _, req_repo, rpt_repo, fill_repo, evt_repo = _make_service()
    plan = _plan(_delta())
    receipts = (_filled_receipt(),)

    await svc.record_execution(plan, receipts, tick_id="tick_001")

    assert len(req_repo.requests) == 1
    assert req_repo.requests[0].instrument_id == 42
    assert req_repo.requests[0].tick_id == "tick_001"

    assert len(rpt_repo.reports) == 1
    assert rpt_repo.reports[0].status == OrderStatus.FILLED

    assert len(fill_repo.fills) == 1
    assert fill_repo.fills[0].price == Decimal("50000")

    assert len(evt_repo.events) == 1
    assert evt_repo.events[0].event_type == OrderEventType.FILLED


@pytest.mark.asyncio
async def test_skipped_receipt_no_fill() -> None:
    svc, _, req_repo, rpt_repo, fill_repo, evt_repo = _make_service()
    plan = _plan(_delta())
    receipts = (ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="no_fill"),)

    await svc.record_execution(plan, receipts, tick_id="tick_002")

    assert len(req_repo.requests) == 1
    assert len(rpt_repo.reports) == 1
    assert rpt_repo.reports[0].status == OrderStatus.CANCELED
    assert len(fill_repo.fills) == 0
    assert len(evt_repo.events) == 1
    assert evt_repo.events[0].event_type == OrderEventType.CANCELED


@pytest.mark.asyncio
async def test_error_receipt_no_fill() -> None:
    svc, _, req_repo, rpt_repo, fill_repo, evt_repo = _make_service()
    plan = _plan(_delta())
    receipts = (ExecutionReceipt(status=ExecutionStatus.ERROR, reason="gateway_error"),)

    await svc.record_execution(plan, receipts, tick_id="tick_003")

    assert len(req_repo.requests) == 1
    assert len(rpt_repo.reports) == 1
    assert rpt_repo.reports[0].status == OrderStatus.REJECTED
    assert len(fill_repo.fills) == 0
    assert len(evt_repo.events) == 1
    assert evt_repo.events[0].event_type == OrderEventType.REJECTED


@pytest.mark.asyncio
async def test_multiple_deltas_mixed_statuses() -> None:
    svc, _, req_repo, _, fill_repo, evt_repo = _make_service()
    plan = _plan(
        _delta(sequence=0, target_index=0),
        _delta(side=OrderSide.SELL, sequence=1, target_index=1, reduce_only=True),
    )
    receipts = (
        _filled_receipt(),
        ExecutionReceipt(status=ExecutionStatus.REJECTED, reason="rejected"),
    )

    await svc.record_execution(plan, receipts, tick_id="tick_004")

    assert len(req_repo.requests) == 2
    assert len(fill_repo.fills) == 1
    assert len(evt_repo.events) == 2
    assert evt_repo.events[0].event_type == OrderEventType.FILLED
    assert evt_repo.events[1].event_type == OrderEventType.REJECTED


@pytest.mark.asyncio
async def test_unknown_instrument_skipped_no_crash() -> None:
    svc, _, req_repo, _, _, _ = _make_service()
    unknown_iid = InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="UNKNOWN",
        base_asset="UNK",
        quote_asset="USDC",
        settle_asset="USDC",
    )
    delta = LegDelta(
        instrument_id=unknown_iid,
        target_index=0,
        position_side=PositionSide.LONG,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
    )
    plan = _plan(delta)
    receipts = (_filled_receipt(),)

    await svc.record_execution(plan, receipts, tick_id="tick_005")

    assert len(req_repo.requests) == 0


@pytest.mark.asyncio
async def test_client_order_id_matches_compute() -> None:
    svc, _, req_repo, _, _, _ = _make_service()
    d = _delta()
    plan = _plan(d)
    receipts = (_filled_receipt(),)
    tick_id = "tick_006"

    await svc.record_execution(plan, receipts, tick_id=tick_id)

    expected_coid = compute_client_order_id(tick_id, "BTCUSDC", d.side, d.position_side, 0, d.order_type)
    assert req_repo.requests[0].client_order_id == expected_coid

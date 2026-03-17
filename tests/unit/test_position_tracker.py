"""Unit tests for PositionTracker."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.domain.enums import (
    ExecutionStatus,
    OrderSide,
    OrderType,
    PositionEventType,
    PositionSide,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.instrument_repository import FakeInstrumentRepo
from fixtures.fakes.position_event_repository import FakePositionEventRepo
from fixtures.fakes.position_leg_repository import FakePositionLegRepo

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_IID = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDC",
    base_asset="BTC",
    quote_asset="USDC",
    settle_asset="USDC",
)
_INSTRUMENT_DB_ID = 42


def _make_tracker() -> tuple[PositionTracker, FakeInstrumentRepo, FakePositionLegRepo, FakePositionEventRepo]:
    instrument_repo = FakeInstrumentRepo()
    instrument_repo.seed(BINANCE_VENUE, BINANCE_USD_C, "BTCUSDC", db_id=_INSTRUMENT_DB_ID)
    leg_repo = FakePositionLegRepo()
    event_repo = FakePositionEventRepo()
    tracker = PositionTracker(
        instrument_repo=instrument_repo,
        position_leg_repo=leg_repo,
        position_event_repo=event_repo,
        clock=FixedClock(_NOW),
    )
    return tracker, instrument_repo, leg_repo, event_repo


def _plan(*deltas: LegDelta) -> ExecutionPlan:
    return ExecutionPlan(account="default", deltas=deltas)


def _delta(
    *,
    side: OrderSide = OrderSide.BUY,
    qty: Decimal = Decimal("0.1"),
    price: Decimal | None = Decimal("50000"),
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
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
        reduce_only=reduce_only,
        close_position=close_position,
        sequence=sequence,
    )


def _filled(
    price: Decimal = Decimal("50000"),
    qty: Decimal = Decimal("0.1"),
) -> ExecutionReceipt:
    return ExecutionReceipt(
        status=ExecutionStatus.FILLED,
        reason="filled",
        fill_price=price,
        filled_quantity=qty,
        order_id="exch_1",
        exchange_timestamp=_NOW,
    )


@pytest.mark.asyncio
async def test_open_new_position_from_flat() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()
    plan = _plan(_delta())
    receipts = (_filled(),)

    await tracker.apply_fills(plan, receipts)

    assert len(leg_repo.legs) == 1
    leg = next(iter(leg_repo.legs.values()))
    assert leg.quantity == Decimal("0.1")
    assert leg.entry_price == Decimal("50000")
    assert leg.position_side == PositionSide.LONG
    assert len(event_repo.events) == 1
    assert event_repo.events[0].event_type is PositionEventType.OPEN


@pytest.mark.asyncio
async def test_add_to_existing_weighted_avg() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()

    # Open initial position
    plan1 = _plan(_delta(qty=Decimal("0.1"), price=Decimal("50000")))
    await tracker.apply_fills(plan1, (_filled(price=Decimal("50000"), qty=Decimal("0.1")),))

    # Add to position at a different price
    plan2 = _plan(_delta(qty=Decimal("0.1"), price=Decimal("52000")))
    await tracker.apply_fills(plan2, (_filled(price=Decimal("52000"), qty=Decimal("0.1")),))

    assert len(leg_repo.legs) == 1
    leg = next(iter(leg_repo.legs.values()))
    assert leg.quantity == Decimal("0.2")
    expected_entry = (Decimal("50000") * Decimal("0.1") + Decimal("52000") * Decimal("0.1")) / Decimal("0.2")
    assert leg.entry_price == expected_entry
    assert len(event_repo.events) == 2
    assert event_repo.events[1].event_type is PositionEventType.ADD


@pytest.mark.asyncio
async def test_reduce_existing_partial_close() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()

    # Open
    await tracker.apply_fills(
        _plan(_delta(qty=Decimal("0.2"))),
        (_filled(price=Decimal("50000"), qty=Decimal("0.2")),),
    )

    # Partial reduce
    await tracker.apply_fills(
        _plan(_delta(side=OrderSide.SELL, qty=Decimal("0.1"), reduce_only=True)),
        (_filled(price=Decimal("52000"), qty=Decimal("0.1")),),
    )

    leg = next(iter(leg_repo.legs.values()))
    assert leg.quantity == Decimal("0.1")
    assert leg.closed_at is None
    assert len(event_repo.events) == 2
    reduce_evt = event_repo.events[1]
    assert reduce_evt.event_type is PositionEventType.REDUCE
    assert reduce_evt.realized_pnl == (Decimal("52000") - Decimal("50000")) * Decimal("0.1")


@pytest.mark.asyncio
async def test_close_to_zero_full_close() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()

    # Open
    await tracker.apply_fills(
        _plan(_delta(qty=Decimal("0.1"))),
        (_filled(price=Decimal("50000"), qty=Decimal("0.1")),),
    )

    # Full close
    await tracker.apply_fills(
        _plan(_delta(side=OrderSide.SELL, qty=Decimal("0.1"), close_position=True)),
        (_filled(price=Decimal("55000"), qty=Decimal("0.1")),),
    )

    leg = next(iter(leg_repo.legs.values()))
    assert leg.closed_at is not None
    assert len(event_repo.events) == 2
    close_evt = event_repo.events[1]
    assert close_evt.event_type is PositionEventType.CLOSE
    assert close_evt.realized_pnl == (Decimal("55000") - Decimal("50000")) * Decimal("0.1")


@pytest.mark.asyncio
async def test_flip_close_plus_open() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()

    # Open long
    await tracker.apply_fills(
        _plan(_delta(qty=Decimal("0.1"), position_side=PositionSide.LONG)),
        (_filled(price=Decimal("50000"), qty=Decimal("0.1")),),
    )

    # Close long + open short (2 deltas)
    close_delta = _delta(
        side=OrderSide.SELL,
        qty=Decimal("0.1"),
        position_side=PositionSide.LONG,
        close_position=True,
        sequence=0,
    )
    open_delta = _delta(
        side=OrderSide.SELL,
        qty=Decimal("0.1"),
        position_side=PositionSide.SHORT,
        sequence=1,
    )
    await tracker.apply_fills(
        _plan(close_delta, open_delta),
        (
            _filled(price=Decimal("48000"), qty=Decimal("0.1")),
            _filled(price=Decimal("48000"), qty=Decimal("0.1")),
        ),
    )

    active_legs = [leg for leg in leg_repo.legs.values() if leg.closed_at is None]
    closed_legs = [leg for leg in leg_repo.legs.values() if leg.closed_at is not None]
    assert len(active_legs) == 1
    assert active_legs[0].position_side == PositionSide.SHORT
    assert len(closed_legs) == 1

    assert event_repo.events[1].event_type is PositionEventType.CLOSE
    assert event_repo.events[2].event_type is PositionEventType.OPEN


@pytest.mark.asyncio
async def test_short_position_pnl() -> None:
    tracker, _, _leg_repo, event_repo = _make_tracker()

    # Open short
    await tracker.apply_fills(
        _plan(_delta(side=OrderSide.SELL, qty=Decimal("0.1"), position_side=PositionSide.SHORT)),
        (_filled(price=Decimal("50000"), qty=Decimal("0.1")),),
    )

    # Close short at lower price (profit)
    await tracker.apply_fills(
        _plan(_delta(side=OrderSide.BUY, qty=Decimal("0.1"), position_side=PositionSide.SHORT, close_position=True)),
        (_filled(price=Decimal("48000"), qty=Decimal("0.1")),),
    )

    close_evt = event_repo.events[1]
    assert close_evt.event_type is PositionEventType.CLOSE
    # SHORT PnL: (entry - fill) * qty = (50000 - 48000) * 0.1 = 200
    assert close_evt.realized_pnl == Decimal("200")


@pytest.mark.asyncio
async def test_skipped_and_error_receipts_ignored() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()
    plan = _plan(
        _delta(sequence=0),
        _delta(sequence=1),
    )
    receipts = (
        ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="skip"),
        ExecutionReceipt(status=ExecutionStatus.ERROR, reason="err"),
    )

    await tracker.apply_fills(plan, receipts)

    assert len(leg_repo.legs) == 0
    assert len(event_repo.events) == 0


@pytest.mark.asyncio
async def test_unknown_instrument_skipped() -> None:
    tracker, _, leg_repo, event_repo = _make_tracker()
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
        quantity=Decimal("0.1"),
    )
    plan = _plan(delta)
    receipts = (_filled(),)

    await tracker.apply_fills(plan, receipts)

    assert len(leg_repo.legs) == 0
    assert len(event_repo.events) == 0

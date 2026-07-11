"""Unit tests for OrderEventConsumer."""

import asyncio
import contextlib
from datetime import UTC, datetime
from decimal import Decimal

from fixtures.fakes.clock import FixedClock
from fixtures.fakes.fill_repository import FakeFillRepo
from fixtures.fakes.instrument_repository import FakeInstrumentRepo
from fixtures.fakes.order_event_repository import FakeOrderEventRepository
from fixtures.fakes.order_event_stream import InMemoryOrderEventStream
from fixtures.fakes.order_report_repository import FakeOrderReportRepo
from fixtures.fakes.order_request_repository import FakeOrderRequestRepo
from fixtures.fakes.position_event_repository import FakePositionEventRepo
from fixtures.fakes.position_leg_repository import FakePositionLegRepo
from fixtures.fakes.stream_cursor_repository import FakeStreamCursorRepo

from dojiwick.application.services.order_event_consumer import OrderEventConsumer
from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.domain.enums import (
    OrderEventType,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_request import OrderRequest

_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_update(
    *,
    client_order_id: str = "order_1",
    exchange_order_id: str = "99001",
    status: OrderStatus = OrderStatus.FILLED,
    execution_type: str = "TRADE",
    last_filled_qty: Decimal = Decimal("0.01"),
    last_filled_price: Decimal = Decimal("95000"),
    cumulative_filled_qty: Decimal = Decimal("0.01"),
    avg_price: Decimal = Decimal("95000"),
    commission: Decimal = Decimal("0.5"),
    trade_id: int = 7777,
    reduce_only: bool = False,
    close_position: bool = False,
) -> ExchangeOrderUpdate:
    return ExchangeOrderUpdate(
        exchange_order_id=exchange_order_id,
        client_order_id=client_order_id,
        symbol="BTCUSDC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        order_status=status,
        execution_type=execution_type,
        position_side=PositionSide.NET,
        last_filled_qty=last_filled_qty,
        last_filled_price=last_filled_price,
        cumulative_filled_qty=cumulative_filled_qty,
        avg_price=avg_price,
        commission=commission,
        commission_asset="USDC",
        trade_id=trade_id,
        order_trade_time=_NOW,
        reduce_only=reduce_only,
        close_position=close_position,
        realized_profit=Decimal(0),
        event_time=_NOW,
        transaction_time=_NOW,
    )


def _build_consumer(
    stream: InMemoryOrderEventStream,
    order_request_repo: FakeOrderRequestRepo | None = None,
    fill_repo: FakeFillRepo | None = None,
    cursor_flush_interval: int = 10,
) -> tuple[
    OrderEventConsumer,
    FakeOrderRequestRepo,
    FakeOrderReportRepo,
    FakeFillRepo,
    FakeOrderEventRepository,
    FakePositionLegRepo,
    FakePositionEventRepo,
    FakeStreamCursorRepo,
]:
    req_repo = order_request_repo or FakeOrderRequestRepo()
    report_repo = FakeOrderReportRepo()
    fr = fill_repo or FakeFillRepo()
    event_repo = FakeOrderEventRepository()
    instrument_repo = FakeInstrumentRepo()
    pos_leg_repo = FakePositionLegRepo()
    pos_event_repo = FakePositionEventRepo()
    cursor_repo = FakeStreamCursorRepo()
    clock = FixedClock()

    position_tracker = PositionTracker(
        instrument_repo=instrument_repo,
        position_leg_repo=pos_leg_repo,
        position_event_repo=pos_event_repo,
        order_request_repo=req_repo,
        clock=clock,
    )

    consumer = OrderEventConsumer(
        stream=stream,
        order_request_repo=req_repo,
        order_report_repo=report_repo,
        fill_repo=fr,
        order_event_repo=event_repo,
        position_tracker=position_tracker,
        cursor_repo=cursor_repo,
        clock=clock,
        cursor_flush_interval=cursor_flush_interval,
    )
    return consumer, req_repo, report_repo, fr, event_repo, pos_leg_repo, pos_event_repo, cursor_repo


async def _seed_order_request(repo: FakeOrderRequestRepo, client_order_id: str = "order_1") -> int:
    """Seed an order request and return its DB id."""
    return await repo.insert_request(
        OrderRequest(
            client_order_id=client_order_id,
            instrument_id=1,
            account="default",
            venue="binance",
            product="usd_c",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            position_side=PositionSide.NET,
        )
    )


async def _drain(consumer: OrderEventConsumer) -> None:
    """Feed the fake stream's updates through the processing path once.

    run() is a reconnect-forever supervisor now; unit tests drive processing
    directly and the supervisor behavior is covered separately.
    """
    try:
        async for update in consumer.stream.raw_updates():
            await consumer.process_update(update)
    finally:
        await consumer.flush_cursor()


async def test_consumer_processes_filled_trade() -> None:
    """Filled TRADE: report upserted, fill inserted, event recorded, cursor set."""
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(_make_update())
    consumer, req_repo, report_repo, fill_repo, event_repo, *_, cursor_repo = _build_consumer(
        stream, cursor_flush_interval=1
    )
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert len(report_repo.reports) == 1
    assert report_repo.reports[0].status is OrderStatus.FILLED
    assert report_repo.reports[0].filled_qty == Decimal("0.01")

    assert len(fill_repo.fills) == 1
    assert fill_repo.fills[0].fill_id == "7777"
    assert fill_repo.fills[0].price == Decimal("95000")

    assert len(event_repo.events) == 1
    assert event_repo.events[0].event_type is OrderEventType.FILLED

    assert "test_orders" in cursor_repo.cursors


async def test_consumer_duplicate_delivery_applies_position_once() -> None:
    """The same trade delivered twice advances the high-water mark only once."""
    update = _make_update()
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    stream.push_raw_update(update)
    consumer, req_repo, _, _fill_repo, _, pos_leg_repo, pos_event_repo, _ = _build_consumer(stream)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert len(pos_leg_repo.legs) == 1
    leg = next(iter(pos_leg_repo.legs.values()))
    assert leg.quantity == Decimal("0.01")
    assert len(pos_event_repo.events) == 1


async def test_consumer_partial_fill_updates_report() -> None:
    """Partial fill: report updated with cumulative_filled_qty, fill inserted."""
    update = _make_update(
        status=OrderStatus.PARTIALLY_FILLED,
        last_filled_qty=Decimal("0.005"),
        cumulative_filled_qty=Decimal("0.005"),
    )
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    consumer, req_repo, report_repo, fill_repo, event_repo, *_ = _build_consumer(stream, cursor_flush_interval=1)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert report_repo.reports[0].status is OrderStatus.PARTIALLY_FILLED
    assert report_repo.reports[0].filled_qty == Decimal("0.005")
    assert len(fill_repo.fills) == 1
    assert event_repo.events[0].event_type is OrderEventType.PARTIALLY_FILLED


async def test_consumer_cancel_event_no_fill() -> None:
    """Cancel event: report status CANCELED, no fill, no position update."""
    update = _make_update(
        status=OrderStatus.CANCELED,
        execution_type="CANCELED",
        last_filled_qty=Decimal(0),
        last_filled_price=Decimal(0),
        cumulative_filled_qty=Decimal(0),
        avg_price=Decimal(0),
        trade_id=0,
    )
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    consumer, req_repo, report_repo, fill_repo, event_repo, pos_leg_repo, *_ = _build_consumer(
        stream, cursor_flush_interval=1
    )
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert report_repo.reports[0].status is OrderStatus.CANCELED
    assert len(fill_repo.fills) == 0
    assert len(pos_leg_repo.legs) == 0
    assert event_repo.events[0].event_type is OrderEventType.CANCELED


async def test_consumer_unknown_order_skipped() -> None:
    """Unknown client_order_id: warning logged, event skipped."""
    update = _make_update(client_order_id="unknown_order")
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    consumer, _, report_repo, fill_repo, event_repo, _, _, cursor_repo = _build_consumer(stream)

    await _drain(consumer)

    assert len(report_repo.reports) == 0
    assert len(fill_repo.fills) == 0
    assert len(event_repo.events) == 0
    assert len(cursor_repo.cursors) == 0


async def test_consumer_cursor_checkpointed() -> None:
    """Cursor is checkpointed after stream ends (via finally flush)."""
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(_make_update())
    consumer, req_repo, *_, cursor_repo = _build_consumer(stream)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    cursor = cursor_repo.cursors.get("test_orders")
    assert cursor is not None
    assert cursor.last_event_id == "99001"
    assert cursor.last_event_time == _NOW


async def test_consumer_new_order_event() -> None:
    """NEW status generates a PLACED event type."""
    update = _make_update(
        status=OrderStatus.NEW,
        execution_type="NEW",
        last_filled_qty=Decimal(0),
        last_filled_price=Decimal(0),
        cumulative_filled_qty=Decimal(0),
        trade_id=0,
    )
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    consumer, req_repo, report_repo, fill_repo, event_repo, *_ = _build_consumer(stream, cursor_flush_interval=1)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert report_repo.reports[0].status is OrderStatus.NEW
    assert len(fill_repo.fills) == 0  # execution_type != "TRADE"
    assert event_repo.events[0].event_type is OrderEventType.PLACED


async def test_cursor_batching_flushes_at_interval() -> None:
    """Cursor is flushed at the interval boundary and on shutdown."""
    updates = [_make_update(exchange_order_id=str(i), trade_id=i) for i in range(1, 4)]
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    for u in updates:
        stream.push_raw_update(u)
    consumer, req_repo, *_, cursor_repo = _build_consumer(stream, cursor_flush_interval=2)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    # After 3 events with interval=2: flush at event 2 (mid-stream), flush at finally (event 3)
    cursor = cursor_repo.cursors.get("test_orders")
    assert cursor is not None
    # Last cursor should be from the 3rd event (flushed in finally)
    assert cursor.last_event_id == "3"


async def test_consumer_passes_realized_pnl_to_fill_and_event() -> None:
    """Exchange-reported realized_profit flows to both Fill and OrderEvent."""
    update = _make_update()
    # Replace with a non-zero realized_profit
    update = ExchangeOrderUpdate(
        exchange_order_id=update.exchange_order_id,
        client_order_id=update.client_order_id,
        symbol=update.symbol,
        side=update.side,
        order_type=update.order_type,
        order_status=update.order_status,
        execution_type=update.execution_type,
        position_side=update.position_side,
        last_filled_qty=update.last_filled_qty,
        last_filled_price=update.last_filled_price,
        cumulative_filled_qty=update.cumulative_filled_qty,
        avg_price=update.avg_price,
        commission=update.commission,
        commission_asset=update.commission_asset,
        trade_id=update.trade_id,
        order_trade_time=update.order_trade_time,
        reduce_only=update.reduce_only,
        close_position=update.close_position,
        realized_profit=Decimal("12.50"),
        event_time=update.event_time,
        transaction_time=update.transaction_time,
    )
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(update)
    consumer, req_repo, _, fill_repo, event_repo, *_ = _build_consumer(stream, cursor_flush_interval=1)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert fill_repo.fills[0].realized_pnl_exchange == Decimal("12.50")
    assert event_repo.events[0].realized_pnl_exchange == Decimal("12.50")


async def test_consumer_propagates_stream_error() -> None:
    """When the stream raises, consumer re-raises after flushing cursor."""
    import pytest

    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(_make_update())
    stream.set_error_after(1, RuntimeError("ws disconnect"))
    consumer, req_repo, *_, cursor_repo = _build_consumer(stream, cursor_flush_interval=1)
    await _seed_order_request(req_repo)

    with pytest.raises(RuntimeError, match="ws disconnect"):
        await _drain(consumer)

    # Cursor was still flushed in the finally block
    cursor = cursor_repo.cursors.get("test_orders")
    assert cursor is not None


async def test_supervisor_reconnects_after_stream_error() -> None:
    """run() survives a stream error: reconnects with backoff instead of dying."""
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(_make_update())
    stream.set_error_after(1, RuntimeError("ws disconnect"))
    consumer, req_repo, report_repo, *_ = _build_consumer(stream, cursor_flush_interval=1)
    consumer.reconnect_base_delay_sec = 0.01
    consumer.reconnect_max_delay_sec = 0.02
    await _seed_order_request(req_repo)

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.1)
    assert not task.done(), "supervisor must keep running through stream errors"
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The event before the error was still processed
    assert len(report_repo.reports) >= 1


async def test_partial_then_full_fill_progression() -> None:
    """Cumulative HWM applies 0.005 then 0.005 more; report ends FILLED."""
    stream = InMemoryOrderEventStream(_stream_name="test_orders")
    stream.push_raw_update(
        _make_update(
            status=OrderStatus.PARTIALLY_FILLED,
            last_filled_qty=Decimal("0.005"),
            cumulative_filled_qty=Decimal("0.005"),
            trade_id=1,
        )
    )
    stream.push_raw_update(
        _make_update(
            status=OrderStatus.FILLED,
            last_filled_qty=Decimal("0.005"),
            cumulative_filled_qty=Decimal("0.010"),
            trade_id=2,
        )
    )
    consumer, req_repo, report_repo, fill_repo, _events, pos_leg_repo, pos_event_repo, _cursor = _build_consumer(stream)
    await _seed_order_request(req_repo)

    await _drain(consumer)

    assert report_repo.reports[-1].status is OrderStatus.FILLED
    assert report_repo.reports[-1].filled_qty == Decimal("0.010")
    assert len(fill_repo.fills) == 2
    leg = next(iter(pos_leg_repo.legs.values()))
    assert leg.quantity == Decimal("0.010"), "high-water mark applies each cumulative delta exactly once"
    assert len(pos_event_repo.events) == 2

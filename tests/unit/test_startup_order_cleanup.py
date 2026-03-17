"""Unit tests for StartupOrderCleanupService."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from dojiwick.application.services.startup_order_cleanup import StartupOrderCleanupService
from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder, OpenOrderPort
from dojiwick.domain.enums import AuditSeverity, OrderEventType, OrderSide, OrderStatus, PositionSide
from dojiwick.domain.models.value_objects.order_request import OrderReport
from fixtures.fakes.audit_log import CapturingAuditLog
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.open_order import FakeOpenOrderAdapter
from fixtures.fakes.order_event_repository import FakeOrderEventRepository
from fixtures.fakes.order_report_repository import FakeOrderReportRepo

_NOW = datetime(2026, 3, 6, tzinfo=UTC)


def _make_service() -> tuple[
    StartupOrderCleanupService,
    FakeOpenOrderAdapter,
    FakeOrderReportRepo,
    FakeOrderEventRepository,
    CapturingAuditLog,
]:
    open_order_adapter = FakeOpenOrderAdapter()
    report_repo = FakeOrderReportRepo()
    event_repo = FakeOrderEventRepository()
    audit_log = CapturingAuditLog()
    svc = StartupOrderCleanupService(
        open_order_port=open_order_adapter,
        order_report_repo=report_repo,
        order_event_repo=event_repo,
        audit_log=audit_log,
        clock=FixedClock(_NOW),
    )
    return svc, open_order_adapter, report_repo, event_repo, audit_log


def _open_order(
    *,
    eid: str = "exch_1",
    cid: str = "dw_abc_123",
    symbol: str = "BTCUSDC",
    side: OrderSide = OrderSide.BUY,
    position_side: PositionSide = PositionSide.LONG,
    status: OrderStatus = OrderStatus.NEW,
    qty: Decimal = Decimal("0.1"),
    filled: Decimal = Decimal(0),
) -> ExchangeOpenOrder:
    return ExchangeOpenOrder(
        exchange_order_id=eid,
        client_order_id=cid,
        symbol=symbol,
        side=side,
        position_side=position_side,
        status=status,
        original_quantity=qty,
        filled_quantity=filled,
    )


@pytest.mark.asyncio
async def test_no_open_orders_clean_result() -> None:
    svc, adapter, _, _, _ = _make_service()

    result = await svc.run(("BTCUSDC",))

    assert result.is_clean
    assert len(result.cancelled) == 0
    assert len(result.errors) == 0
    assert len(adapter.cancel_calls) == 0


@pytest.mark.asyncio
async def test_stale_orders_cancelled() -> None:
    svc, adapter, _, _, _ = _make_service()
    adapter.seed(
        "BTCUSDC",
        [
            _open_order(eid="exch_1"),
            _open_order(eid="exch_2", cid="dw_abc_456"),
        ],
    )

    result = await svc.run(("BTCUSDC",))

    assert len(result.cancelled) == 2
    assert result.cancelled[0].exchange_order_id == "exch_1"
    assert result.cancelled[1].exchange_order_id == "exch_2"
    assert "BTCUSDC" in adapter.cancel_calls


@pytest.mark.asyncio
async def test_db_report_updated_to_canceled() -> None:
    svc, adapter, report_repo, event_repo, _ = _make_service()
    adapter.seed("BTCUSDC", [_open_order(eid="exch_1")])
    report_repo.reports.append(
        OrderReport(
            order_request_id=10,
            exchange_order_id="exch_1",
            status=OrderStatus.NEW,
            filled_qty=Decimal(0),
            reported_at=_NOW,
            id=1,
        )
    )

    await svc.run(("BTCUSDC",))

    updated = await report_repo.get_by_exchange_order_id("exch_1")
    assert updated is not None
    assert updated.status == OrderStatus.CANCELED

    assert len(event_repo.events) == 1
    assert event_repo.events[0].event_type == OrderEventType.CANCELED
    assert event_repo.events[0].detail == "startup_cleanup"


@pytest.mark.asyncio
async def test_open_order_not_in_db_logged() -> None:
    svc, adapter, report_repo, _, _ = _make_service()
    adapter.seed("BTCUSDC", [_open_order(eid="exch_missing")])

    result = await svc.run(("BTCUSDC",))

    assert len(result.cancelled) == 1
    assert len(result.errors) == 0
    assert len(report_repo.reports) == 0


@pytest.mark.asyncio
async def test_multiple_symbols_processed() -> None:
    svc, adapter, _, _, _ = _make_service()
    adapter.seed("ETHUSDC", [_open_order(eid="exch_eth", symbol="ETHUSDC")])

    result = await svc.run(("BTCUSDC", "ETHUSDC", "SOLUSDC"))

    assert len(result.cancelled) == 1
    assert result.cancelled[0].symbol == "ETHUSDC"
    assert "ETHUSDC" in adapter.cancel_calls
    assert "BTCUSDC" not in adapter.cancel_calls


@pytest.mark.asyncio
async def test_exchange_error_recorded() -> None:
    """Simulate exchange error via a raising adapter."""
    from dojiwick.domain.errors import AdapterError

    report_repo = FakeOrderReportRepo()
    event_repo = FakeOrderEventRepository()
    audit_log = CapturingAuditLog()

    class ErrorAdapter(OpenOrderPort):
        async def get_open_orders(self, symbol: str) -> tuple[ExchangeOpenOrder, ...]:
            raise AdapterError(f"timeout for {symbol}")

        async def cancel_all_open_orders(self, symbol: str) -> None:
            pass

    svc = StartupOrderCleanupService(
        open_order_port=ErrorAdapter(),
        order_report_repo=report_repo,
        order_event_repo=event_repo,
        audit_log=audit_log,
        clock=FixedClock(_NOW),
    )

    result = await svc.run(("BTCUSDC",))

    assert len(result.errors) == 1
    assert "timeout" in result.errors[0]


@pytest.mark.asyncio
async def test_audit_log_written() -> None:
    svc, adapter, _, _, audit_log = _make_service()
    adapter.seed("BTCUSDC", [_open_order(eid="exch_1")])

    await svc.run(("BTCUSDC",))

    assert len(audit_log.events) == 1
    event = audit_log.events[0]
    assert event["event_type"] == "startup_order_cleanup"
    assert event["severity"] == AuditSeverity.WARNING
    context = event["context"]
    assert isinstance(context, dict)
    assert context["cancelled_count"] == 1

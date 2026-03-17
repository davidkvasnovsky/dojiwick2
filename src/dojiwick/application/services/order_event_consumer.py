"""Async consumer for real-time WebSocket order events.

Processes ORDER_TRADE_UPDATE events from the exchange user-data stream,
keeping order reports, fills, audit events, and position legs up-to-date
continuously between ticks.
"""

import logging
from dataclasses import dataclass, field

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.order_event_stream import OrderEventStreamPort
from dojiwick.domain.contracts.repositories.fill import FillRepositoryPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.contracts.repositories.stream_cursor import StreamCursorRepositoryPort
from dojiwick.domain.enums import STATUS_TO_EVENT_TYPE
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.order_request import Fill, OrderReport
from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord

from dojiwick.application.services.position_tracker import PositionTracker

log = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderEventConsumer:
    """Consumes WS order updates and persists reports, fills, events, positions."""

    stream: OrderEventStreamPort
    order_request_repo: OrderRequestRepositoryPort
    order_report_repo: OrderReportRepositoryPort
    fill_repo: FillRepositoryPort
    order_event_repo: OrderEventRepositoryPort
    position_tracker: PositionTracker
    cursor_repo: StreamCursorRepositoryPort
    clock: ClockPort
    cursor_flush_interval: int = 10

    _pending_cursor: StreamCursorRecord | None = field(default=None, init=False)
    _events_since_flush: int = field(default=0, init=False)

    async def run(self) -> None:
        """Main async loop — process WS updates until cancelled."""
        try:
            async for update in self.stream.raw_updates():
                await self._process_update(update)
        finally:
            await self._flush_cursor()

    async def _process_update(self, update: ExchangeOrderUpdate) -> None:
        """Process a single ExchangeOrderUpdate."""
        order_request = await self.order_request_repo.get_by_client_order_id(update.client_order_id)
        if order_request is None:
            log.warning("unknown client_order_id=%s — skipping WS event", update.client_order_id)
            return

        assert order_request.id is not None
        order_request_id = order_request.id
        instrument_id = order_request.instrument_id
        account = order_request.account

        event_type = STATUS_TO_EVENT_TYPE.get(update.order_status)
        if event_type is not None:
            await self.order_event_repo.record_event(
                OrderEvent(
                    order_id=order_request_id,
                    event_type=event_type,
                    occurred_at=update.event_time,
                    exchange_order_id=update.exchange_order_id,
                    filled_quantity=update.last_filled_qty,
                    fees_usd=update.commission,
                    realized_pnl_exchange=update.realized_profit,
                )
            )

        await self.order_report_repo.upsert_report(
            OrderReport(
                order_request_id=order_request_id,
                exchange_order_id=update.exchange_order_id,
                status=update.order_status,
                filled_qty=update.cumulative_filled_qty,
                avg_price=update.avg_price if update.avg_price > 0 else None,
                reported_at=update.event_time,
            )
        )

        if update.execution_type == "TRADE":
            fill_id = await self.fill_repo.insert_fill(
                Fill(
                    order_request_id=order_request_id,
                    price=update.last_filled_price,
                    quantity=update.last_filled_qty,
                    fill_id=str(update.trade_id),
                    commission=update.commission,
                    commission_asset=update.commission_asset,
                    realized_pnl_exchange=update.realized_profit,
                    filled_at=update.order_trade_time,
                )
            )
            if fill_id is not None:
                is_decreasing = order_request.reduce_only or order_request.close_position
                await self.position_tracker.update_from_fill(
                    account=account,
                    instrument_id=instrument_id,
                    position_side=order_request.position_side,
                    fill_qty=update.last_filled_qty,
                    fill_price=update.last_filled_price,
                    is_decreasing=is_decreasing,
                )

        self._pending_cursor = StreamCursorRecord(
            stream_name=self.stream.stream_name,
            last_event_id=update.exchange_order_id,
            last_event_time=update.event_time,
        )
        self._events_since_flush += 1
        if self._events_since_flush >= self.cursor_flush_interval:
            await self._flush_cursor()

    async def _flush_cursor(self) -> None:
        """Persist the pending cursor and reset the counter."""
        if self._pending_cursor is not None:
            await self.cursor_repo.set_cursor(self._pending_cursor)
            self._pending_cursor = None
            self._events_since_flush = 0

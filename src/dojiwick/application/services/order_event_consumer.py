"""Async consumer for real-time WebSocket order events.

Processes ORDER_TRADE_UPDATE events from the exchange user-data stream,
keeping order reports, fills, audit events, and position legs up-to-date
continuously between ticks.

The run loop is a supervisor: a dropped socket reconnects with exponential
backoff instead of silently ending the task and halting the engine, and
every (re)connect runs a per-symbol REST recovery sweep
so fills that happened while the socket was down are applied through the
same idempotent high-water-mark path as live events.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.application.services.protective_orders import ProtectiveOrderService
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.order_event_stream import OrderEventStreamPort
from dojiwick.domain.contracts.repositories.fill import FillRepositoryPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.contracts.repositories.stream_cursor import StreamCursorRepositoryPort
from dojiwick.domain.enums import PROTECTIVE_ORDER_KINDS, STATUS_TO_EVENT_TYPE
from dojiwick.domain.errors import AdapterError, AuthenticationError
from dojiwick.domain.models.value_objects.exchange_order_update import ExchangeOrderUpdate
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.order_request import Fill, OrderReport
from dojiwick.domain.models.value_objects.stream_cursor_record import StreamCursorRecord

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
    protective_orders: ProtectiveOrderService | None = None
    reconnect_base_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 60.0
    cursor_flush_interval: int = 10

    _pending_cursor: StreamCursorRecord | None = field(default=None, init=False)
    _events_since_flush: int = field(default=0, init=False)

    async def run(self) -> None:
        """Supervised consume loop — reconnects on stream failure until cancelled."""
        attempt = 0
        try:
            while True:
                try:
                    if not self.stream.is_connected:
                        await self.stream.connect()
                        attempt = 0
                    async for update in self.stream.raw_updates():
                        try:
                            await self.process_update(update)
                        except AdapterError:
                            # Persistence is down — reconnecting won't help;
                            # let the runner's task watchdog halt the engine
                            raise
                        except Exception:
                            # One malformed event must not kill order tracking;
                            # the recovery sweep and reconciliation are the net
                            log.exception("failed to process order update: %s", update.client_order_id)
                    log.warning("order event stream ended")
                except asyncio.CancelledError:
                    raise
                except AdapterError, AuthenticationError:
                    raise
                except Exception:
                    log.warning("order event stream error", exc_info=True)

                attempt += 1
                delay = min(self.reconnect_base_delay_sec * 2 ** (attempt - 1), self.reconnect_max_delay_sec)
                log.info("reconnecting order stream in %.1fs (attempt %d)", delay, attempt)
                await asyncio.sleep(delay)
                try:
                    await self.stream.disconnect()
                    await self.stream.connect()
                except Exception:
                    log.warning("order stream reconnect failed", exc_info=True)
        finally:
            await self.flush_cursor()

    async def process_update(self, update: ExchangeOrderUpdate) -> None:
        """Process a single ExchangeOrderUpdate."""
        order_request = await self.order_request_repo.get_by_client_order_id(update.client_order_id)
        if order_request is None:
            # Requests are pre-persisted before submission, so this is either
            # a manual/foreign order or a request row lost to a crash
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
            await self.fill_repo.insert_fill(
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
            is_decreasing = order_request.reduce_only or order_request.close_position
            applied = await self.position_tracker.apply_order_fill(
                order_request_id=order_request_id,
                cumulative_filled_qty=update.cumulative_filled_qty,
                fill_price=update.last_filled_price,
                account=account,
                instrument_id=instrument_id,
                position_side=order_request.position_side,
                is_decreasing=is_decreasing,
            )
            if (
                applied > 0
                and self.protective_orders is not None
                and order_request.order_kind in PROTECTIVE_ORDER_KINDS
                and order_request.position_leg_id is not None
            ):
                await self._handle_protective_fill(order_request.position_leg_id, update.symbol)

        self._pending_cursor = StreamCursorRecord(
            stream_name=self.stream.stream_name,
            last_event_id=update.exchange_order_id,
            last_event_time=update.event_time,
        )
        self._events_since_flush += 1
        if self._events_since_flush >= self.cursor_flush_interval:
            await self.flush_cursor()

    async def _handle_protective_fill(self, position_leg_id: int, symbol: str) -> None:
        """A protective order fired: cancel the surviving sibling when the leg closed.

        The consumer cancels only — placements happen in the tick's sync pass,
        so the two writers never race on order creation.
        """
        assert self.protective_orders is not None
        leg = await self.position_tracker.position_leg_repo.get_leg(position_leg_id)
        if leg is None or leg.closed_at is not None or leg.quantity <= 0:
            await self.protective_orders.on_leg_closed(position_leg_id, symbol)

    async def flush_cursor(self) -> None:
        """Persist the pending cursor and reset the counter.

        Public so replay callers (startup orchestrator) can flush a batch
        smaller than ``cursor_flush_interval`` — otherwise the same events
        replay again on every restart.
        """
        if self._pending_cursor is not None:
            await self.cursor_repo.set_cursor(self._pending_cursor)
            self._pending_cursor = None
            self._events_since_flush = 0

"""Order ledger service — records execution results to DB."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.enums import OrderKind, ExecutionStatus, OrderEventType, OrderStatus
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.order_request import OrderReport, OrderRequest
from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt

log = logging.getLogger(__name__)

_STATUS_MAP: dict[ExecutionStatus, OrderStatus] = {
    ExecutionStatus.FILLED: OrderStatus.FILLED,
    ExecutionStatus.REJECTED: OrderStatus.REJECTED,
    ExecutionStatus.SKIPPED: OrderStatus.CANCELED,
    ExecutionStatus.ERROR: OrderStatus.REJECTED,
    ExecutionStatus.CANCELLED: OrderStatus.CANCELED,
}

_EVENT_MAP: dict[ExecutionStatus, OrderEventType] = {
    ExecutionStatus.FILLED: OrderEventType.FILLED,
    ExecutionStatus.REJECTED: OrderEventType.REJECTED,
    ExecutionStatus.SKIPPED: OrderEventType.CANCELED,
    ExecutionStatus.ERROR: OrderEventType.REJECTED,
    ExecutionStatus.CANCELLED: OrderEventType.CANCELED,
}


@dataclass(slots=True)
class OrderLedgerService:
    """Persists order requests, reports, fills, and events from execution results."""

    instrument_repo: InstrumentRepositoryPort
    order_request_repo: OrderRequestRepositoryPort
    order_report_repo: OrderReportRepositoryPort
    order_event_repo: OrderEventRepositoryPort
    clock: ClockPort

    async def record_requests(
        self,
        plan: ExecutionPlan,
        *,
        tick_id: str,
    ) -> dict[int, int]:
        """Pre-persist order requests BEFORE execution, keyed by delta index.

        The WS ORDER_TRADE_UPDATE usually arrives before the tick's
        post-execution transaction commits; without the request row the
        consumer drops the event as unknown and the fill is lost. Committing
        the intent first also closes the placed-but-crash window: startup
        finds a request with no report and reconciles it against the exchange.
        """
        indexed_deltas = sorted(enumerate(plan.deltas), key=lambda p: p[1].sequence)
        resolved_ids: dict[tuple[str, str, str], int | None] = {}
        request_ids: dict[int, int] = {}

        for leg_seq, (original_index, delta) in enumerate(indexed_deltas):
            iid = delta.instrument_id
            cache_key = (iid.venue, iid.product, iid.symbol)
            if cache_key not in resolved_ids:
                resolved_ids[cache_key] = await self.instrument_repo.resolve_id(iid.venue, iid.product, iid.symbol)
            instrument_id_int = resolved_ids[cache_key]
            if instrument_id_int is None:
                raise AdapterError(f"unknown instrument {iid.venue}/{iid.product}/{iid.symbol} for planned order")

            client_order_id = compute_client_order_id(
                tick_id, iid.symbol, delta.side, delta.position_side, leg_seq, delta.order_type
            )
            kind = OrderKind.EXIT if (delta.reduce_only or delta.close_position) else OrderKind.ENTRY
            request = OrderRequest(
                client_order_id=client_order_id,
                instrument_id=instrument_id_int,
                account=plan.account,
                venue=iid.venue,
                product=iid.product,
                tick_id=tick_id,
                side=delta.side,
                order_type=delta.order_type,
                quantity=delta.quantity,
                price=delta.price,
                position_side=delta.position_side,
                reduce_only=delta.reduce_only,
                close_position=delta.close_position,
                time_in_force=delta.time_in_force,
                working_type=delta.working_type,
                order_kind=kind,
            )
            request_ids[original_index] = await self.order_request_repo.insert_request(request)

        return request_ids

    async def record_results(
        self,
        plan: ExecutionPlan,
        plan_receipts: tuple[ExecutionReceipt, ...],
        request_ids: dict[int, int],
    ) -> None:
        """Record reports and lifecycle events for executed deltas.

        Fill rows are written exclusively by the WS consumer (trade-id keyed);
        writing them here too created undeduplicatable blank-id duplicates.
        """
        for original_index, delta in enumerate(plan.deltas):
            _ = delta
            request_id = request_ids.get(original_index)
            if request_id is None:
                continue
            receipt = (
                plan_receipts[original_index]
                if original_index < len(plan_receipts)
                else ExecutionReceipt(status=ExecutionStatus.ERROR, reason="missing_receipt")
            )

            order_status = _STATUS_MAP.get(receipt.status, OrderStatus.REJECTED)
            exchange_order_id = receipt.order_id or f"none_{request_id}"
            now = self.clock.now_utc()

            report = OrderReport(
                order_request_id=request_id,
                exchange_order_id=exchange_order_id,
                status=order_status,
                filled_qty=receipt.filled_quantity,
                avg_price=receipt.fill_price,
                reported_at=receipt.exchange_timestamp or now,
            )
            await self.order_report_repo.upsert_report(report)

            event_type = _EVENT_MAP.get(receipt.status, OrderEventType.REJECTED)
            is_filled = receipt.status is ExecutionStatus.FILLED
            event = OrderEvent(
                order_id=request_id,
                event_type=event_type,
                occurred_at=receipt.exchange_timestamp or now,
                exchange_order_id=exchange_order_id,
                filled_quantity=receipt.filled_quantity if is_filled else Decimal(0),
                fees_usd=receipt.fees_usd if is_filled else Decimal(0),
                fee_asset=receipt.fee_asset if is_filled else "",
                native_fee_amount=receipt.native_fee_amount if is_filled else Decimal(0),
            )
            await self.order_event_repo.record_event(event)

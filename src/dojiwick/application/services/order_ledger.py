"""Order ledger service — records execution results to DB."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.repositories.fill import FillRepositoryPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.order_event import OrderEventRepositoryPort
from dojiwick.domain.contracts.repositories.order_report import OrderReportRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.enums import ExecutionStatus, OrderEventType, OrderStatus
from dojiwick.domain.hashing import compute_client_order_id
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.order_request import Fill, OrderReport, OrderRequest
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
    fill_repo: FillRepositoryPort
    order_event_repo: OrderEventRepositoryPort
    clock: ClockPort

    async def record_execution(
        self,
        plan: ExecutionPlan,
        plan_receipts: tuple[ExecutionReceipt, ...],
        *,
        tick_id: str,
    ) -> None:
        """Record all order lifecycle data from an execution plan and its receipts."""
        indexed_deltas = sorted(enumerate(plan.deltas), key=lambda p: p[1].sequence)
        resolved_ids: dict[tuple[str, str, str], int | None] = {}

        for leg_seq, (original_index, delta) in enumerate(indexed_deltas):
            receipt = (
                plan_receipts[original_index]
                if original_index < len(plan_receipts)
                else ExecutionReceipt(status=ExecutionStatus.ERROR, reason="missing_receipt")
            )
            iid = delta.instrument_id
            cache_key = (iid.venue, iid.product, iid.symbol)
            if cache_key not in resolved_ids:
                resolved_ids[cache_key] = await self.instrument_repo.resolve_id(iid.venue, iid.product, iid.symbol)
            instrument_id_int = resolved_ids[cache_key]
            if instrument_id_int is None:
                log.warning("unknown instrument %s/%s/%s — skipping ledger entry", iid.venue, iid.product, iid.symbol)
                continue

            client_order_id = compute_client_order_id(
                tick_id, iid.symbol, delta.side, delta.position_side, leg_seq, delta.order_type
            )

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
            )
            request_id = await self.order_request_repo.insert_request(request)

            order_status = _STATUS_MAP.get(receipt.status, OrderStatus.REJECTED)
            exchange_order_id = receipt.order_id or f"none_{client_order_id}"
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

            if receipt.status is ExecutionStatus.FILLED and receipt.fill_price is not None:
                fill = Fill(
                    order_request_id=request_id,
                    price=receipt.fill_price,
                    quantity=receipt.filled_quantity,
                    commission=receipt.native_fee_amount if receipt.native_fee_amount else receipt.fees_usd,
                    commission_asset=receipt.fee_asset,
                    filled_at=receipt.exchange_timestamp or now,
                )
                await self.fill_repo.insert_fill(fill)

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

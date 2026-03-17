"""Deterministic hashing for tick identity and content-addressable pipeline stages."""

from __future__ import annotations

import hashlib
from datetime import datetime

from dojiwick.domain.enums import OrderSide, OrderType, PositionSide
from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext, BatchExecutionIntent
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan


def compute_tick_id(config_hash: str, observed_at: datetime, pairs: tuple[str, ...]) -> str:
    """Deterministic tick identity from config, time, and pair set.

    Pair-order-invariant via ``sorted()``.  Returns first 16 hex chars
    (64-bit, collision-safe for practical tick volumes).
    """
    sorted_pairs = ",".join(sorted(pairs))
    payload = f"{config_hash}|{observed_at.isoformat()}|{sorted_pairs}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compute_inputs_hash(context: BatchDecisionContext) -> str:
    """Content-addressable hash of the decision surface inputs.

    Covers all data that feeds the deterministic pipeline *before*
    regime classification (regimes are derived, not an input).
    """
    h = hashlib.sha256()
    h.update(",".join(sorted(context.market.pairs)).encode())
    h.update(context.market.observed_at.isoformat().encode())
    h.update(context.market.price.tobytes())
    h.update(context.market.indicators.tobytes())
    h.update(context.portfolio.equity_usd.tobytes())
    h.update(context.portfolio.day_start_equity_usd.tobytes())
    h.update(context.portfolio.open_positions_total.tobytes())
    h.update(context.portfolio.has_open_position.tobytes())
    h.update(context.portfolio.unrealized_pnl_usd.tobytes())
    return h.hexdigest()[:16]


def compute_intent_hash(intents: BatchExecutionIntent) -> str:
    """Content-addressable hash of the execution intent vectors."""
    h = hashlib.sha256()
    h.update(intents.action.tobytes())
    h.update(intents.entry_price.tobytes())
    h.update(intents.stop_price.tobytes())
    h.update(intents.take_profit_price.tobytes())
    h.update(intents.quantity.tobytes())
    h.update(intents.active_mask.tobytes())
    return h.hexdigest()[:16]


def compute_ops_hash(plan: ExecutionPlan | None) -> str:
    """Content-addressable hash of the execution plan deltas."""
    if plan is None or plan.is_empty:
        return hashlib.sha256(b"EMPTY_PLAN").hexdigest()[:16]

    h = hashlib.sha256()
    for delta in plan.deltas:
        h.update(delta.instrument_id.symbol.encode())
        h.update(delta.instrument_id.venue.encode())
        h.update(str(delta.target_index).encode())
        h.update(delta.position_side.value.encode())
        h.update(delta.side.value.encode())
        h.update(delta.order_type.value.encode())
        h.update(str(delta.quantity).encode())
        h.update(str(delta.price).encode())
        h.update(str(delta.reduce_only).encode())
        h.update(str(delta.close_position).encode())
    return h.hexdigest()[:16]


def compute_client_order_id(
    tick_id: str,
    symbol: str,
    side: OrderSide,
    position_side: PositionSide,
    leg_seq: int,
    op_type: OrderType,
) -> str:
    """Deterministic client order ID for exchange idempotency.

    Format: ``dw_{tick_id[:8]}_{sha256[:12]}`` — 24 chars, within typical exchange 36-char limit.
    """
    payload = f"{tick_id}|{symbol}|{side}|{position_side}|{leg_seq}|{op_type}"
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"dw_{tick_id[:8]}_{suffix}"

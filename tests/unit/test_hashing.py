"""Unit tests for deterministic hashing functions."""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np

from dojiwick.domain.enums import OrderSide, OrderType, PositionSide
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.hashing import (
    compute_client_order_id,
    compute_inputs_hash,
    compute_intent_hash,
    compute_ops_hash,
    compute_tick_id,
)
from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId


def _make_context(
    pairs: tuple[str, ...] = ("BTC/USDC", "ETH/USDC"),
    observed_at: datetime | None = None,
) -> BatchDecisionContext:
    size = len(pairs)
    at = observed_at or datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return BatchDecisionContext(
        market=BatchMarketSnapshot(
            pairs=pairs,
            observed_at=at,
            price=np.full(size, 100.0, dtype=np.float64),
            indicators=np.full((size, INDICATOR_COUNT), 50.0, dtype=np.float64),
        ),
        portfolio=BatchPortfolioSnapshot(
            equity_usd=np.full(size, 1000.0, dtype=np.float64),
            day_start_equity_usd=np.full(size, 1000.0, dtype=np.float64),
            open_positions_total=np.zeros(size, dtype=np.int64),
            has_open_position=np.zeros(size, dtype=np.bool_),
            unrealized_pnl_usd=np.zeros(size, dtype=np.float64),
        ),
    )


def _make_intents(size: int = 2) -> BatchExecutionIntent:
    return BatchExecutionIntent(
        pairs=("BTC/USDC", "ETH/USDC")[:size],
        action=np.array([1, 0][:size], dtype=np.int64),
        quantity=np.full(size, 0.01, dtype=np.float64),
        notional_usd=np.full(size, 100.0, dtype=np.float64),
        entry_price=np.full(size, 100.0, dtype=np.float64),
        stop_price=np.full(size, 95.0, dtype=np.float64),
        take_profit_price=np.full(size, 110.0, dtype=np.float64),
        strategy_name=("trend_follow",) * size,
        strategy_variant=("baseline",) * size,
        active_mask=np.ones(size, dtype=np.bool_),
    )


def _make_plan() -> ExecutionPlan:
    instrument = InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="BTCUSDC",
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )
    return ExecutionPlan(
        account="default",
        deltas=(
            LegDelta(
                instrument_id=instrument,
                target_index=0,
                position_side=PositionSide.NET,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.01"),
            ),
        ),
    )


# --- compute_tick_id ---


def test_tick_id_deterministic() -> None:
    at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    pairs = ("BTC/USDC", "ETH/USDC")
    a = compute_tick_id("cfg1", at, pairs)
    b = compute_tick_id("cfg1", at, pairs)
    assert a == b


def test_tick_id_sensitive_to_time() -> None:
    pairs = ("BTC/USDC",)
    a = compute_tick_id("cfg1", datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC), pairs)
    b = compute_tick_id("cfg1", datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC), pairs)
    assert a != b


def test_tick_id_sensitive_to_config() -> None:
    at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    pairs = ("BTC/USDC",)
    a = compute_tick_id("cfg1", at, pairs)
    b = compute_tick_id("cfg2", at, pairs)
    assert a != b


def test_tick_id_sensitive_to_pairs() -> None:
    at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    a = compute_tick_id("cfg1", at, ("BTC/USDC",))
    b = compute_tick_id("cfg1", at, ("ETH/USDC",))
    assert a != b


def test_tick_id_pair_order_invariant() -> None:
    at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    a = compute_tick_id("cfg1", at, ("BTC/USDC", "ETH/USDC"))
    b = compute_tick_id("cfg1", at, ("ETH/USDC", "BTC/USDC"))
    assert a == b


def test_tick_id_length() -> None:
    at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    result = compute_tick_id("cfg1", at, ("BTC/USDC",))
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


# --- compute_inputs_hash ---


def test_inputs_hash_deterministic() -> None:
    ctx = _make_context()
    a = compute_inputs_hash(ctx)
    b = compute_inputs_hash(ctx)
    assert a == b
    assert len(a) == 16


# --- compute_intent_hash ---


def test_intent_hash_deterministic() -> None:
    intents = _make_intents()
    a = compute_intent_hash(intents)
    b = compute_intent_hash(intents)
    assert a == b
    assert len(a) == 16


# --- compute_ops_hash ---


def test_ops_hash_deterministic() -> None:
    plan = _make_plan()
    a = compute_ops_hash(plan)
    b = compute_ops_hash(plan)
    assert a == b
    assert len(a) == 16


def test_ops_hash_none_plan() -> None:
    a = compute_ops_hash(None)
    b = compute_ops_hash(None)
    assert a == b
    assert len(a) == 16


def test_ops_hash_empty_plan() -> None:
    plan = ExecutionPlan(account="default", deltas=())
    a = compute_ops_hash(plan)
    b = compute_ops_hash(None)
    assert a == b


# --- compute_client_order_id ---


def test_client_order_id_deterministic() -> None:
    a = compute_client_order_id("abc123def456", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
    b = compute_client_order_id("abc123def456", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
    assert a == b


def test_client_order_id_length() -> None:
    result = compute_client_order_id(
        "abc123def456ghij", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET
    )
    assert len(result) <= 36


def test_client_order_id_unique_per_leg() -> None:
    a = compute_client_order_id("abc123def456", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
    b = compute_client_order_id("abc123def456", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 1, OrderType.MARKET)
    assert a != b


def test_client_order_id_format() -> None:
    result = compute_client_order_id("abc123def456", "BTCUSDC", OrderSide.BUY, PositionSide.NET, 0, OrderType.MARKET)
    assert result.startswith("dw_")
    parts = result.split("_")
    assert len(parts) == 3
    assert all(c in "0123456789abcdef" for c in parts[2])

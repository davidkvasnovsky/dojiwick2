"""Unit tests for DefaultExecutionPlanner — mandatory delta scenarios."""

from decimal import Decimal

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.domain.enums import (
    OrderSide,
    OrderType,
    PositionMode,
    PositionSide,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.account_state import (
    AccountBalance,
    AccountSnapshot,
    ExchangePositionLeg,
)
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, TargetLegPosition

BTCUSDC = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDC",
    base_asset="BTC",
    quote_asset="USDC",
    settle_asset="USDC",
)


def _snapshot(
    positions: tuple[ExchangePositionLeg, ...] = (),
    account: str = "default",
) -> AccountSnapshot:
    return AccountSnapshot(
        account=account,
        balances=(AccountBalance(asset="USDC", wallet_balance=Decimal(10_000), available_balance=Decimal(5_000)),),
        positions=positions,
        total_wallet_balance=Decimal(10_000),
        available_balance=Decimal(5_000),
    )


def _long_leg(qty: Decimal, side: PositionSide = PositionSide.NET) -> ExchangePositionLeg:
    return ExchangePositionLeg(
        instrument_id=BTCUSDC,
        position_side=side,
        quantity=qty,
        entry_price=Decimal(50_000),
        unrealized_pnl=Decimal(0),
    )


# Scenario 1: 0.3 long → 0.5 long => buy 0.2


async def test_increase_long_one_way() -> None:
    """0.3 BTC long -> 0.5 BTC long = buy 0.2 BTC (one-way mode)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("0.3")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)

    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.BUY
    assert delta.quantity == Decimal("0.2")
    assert delta.order_type == OrderType.MARKET
    assert delta.position_side == PositionSide.NET
    assert delta.reduce_only is False


# Scenario 2: 0.3 long → 0.1 short => close long then open short


async def test_flip_long_to_short_one_way() -> None:
    """0.3 BTC long -> 0.1 BTC short = close long (sell 0.3) + open short (sell 0.1)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("0.3")),))
    # In one-way mode, target_resolver emits NET with signed qty (negative = short)
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("-0.1"),
        ),
    )

    plan = await planner.plan(snapshot, targets)

    assert len(plan.deltas) == 2

    # First delta: close long (sequence=0)
    close_delta = plan.deltas[0]
    assert close_delta.side == OrderSide.SELL
    assert close_delta.quantity == Decimal("0.3")
    assert close_delta.reduce_only is True
    assert close_delta.close_position is True
    assert close_delta.sequence == 0

    # Second delta: open short (sequence=1)
    open_delta = plan.deltas[1]
    assert open_delta.side == OrderSide.SELL
    assert open_delta.quantity == Decimal("0.1")
    assert open_delta.reduce_only is False
    assert open_delta.sequence == 1


# Scenario 3: Hedge dual-leg targets with independent LONG/SHORT adjustments


async def test_hedge_dual_leg_independent() -> None:
    """Hedge mode: independent LONG and SHORT adjustments on same instrument."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.HEDGE)

    long_leg = ExchangePositionLeg(
        instrument_id=BTCUSDC,
        position_side=PositionSide.LONG,
        quantity=Decimal("0.2"),
        entry_price=Decimal(50_000),
        unrealized_pnl=Decimal(0),
    )
    short_leg = ExchangePositionLeg(
        instrument_id=BTCUSDC,
        position_side=PositionSide.SHORT,
        quantity=Decimal("0.1"),
        entry_price=Decimal(51_000),
        unrealized_pnl=Decimal(0),
    )
    snapshot = _snapshot(positions=(long_leg, short_leg))

    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.LONG,
            target_qty=Decimal("0.5"),
        ),
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.SHORT,
            target_qty=Decimal("0.3"),
        ),
    )

    plan = await planner.plan(snapshot, targets)

    assert len(plan.deltas) == 2

    long_delta = plan.deltas[0]
    assert long_delta.position_side == PositionSide.LONG
    assert long_delta.side == OrderSide.BUY
    assert long_delta.quantity == Decimal("0.3")  # 0.5 - 0.2

    short_delta = plan.deltas[1]
    assert short_delta.position_side == PositionSide.SHORT
    assert short_delta.side == OrderSide.SELL
    assert short_delta.quantity == Decimal("0.2")  # 0.3 - 0.1


# Additional scenarios


async def test_no_change_produces_empty_plan() -> None:
    """Current state already matches target — empty plan."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("0.5")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert plan.is_empty


async def test_open_from_zero() -> None:
    """No current position → open new long."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot()
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    assert plan.deltas[0].side == OrderSide.BUY
    assert plan.deltas[0].quantity == Decimal("0.5")


async def test_decrease_long_one_way() -> None:
    """0.5 long → 0.2 long = sell 0.3 (reduce_only)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("0.5")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0.2"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.SELL
    assert delta.quantity == Decimal("0.3")
    assert delta.reduce_only is True


async def test_hedge_close_short_leg() -> None:
    """Hedge mode: close SHORT leg entirely (target=0)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.HEDGE)
    short_leg = ExchangePositionLeg(
        instrument_id=BTCUSDC,
        position_side=PositionSide.SHORT,
        quantity=Decimal("0.3"),
        entry_price=Decimal(50_000),
        unrealized_pnl=Decimal(0),
    )
    snapshot = _snapshot(positions=(short_leg,))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.SHORT,
            target_qty=Decimal("0"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.BUY
    assert delta.quantity == Decimal("0.3")
    assert delta.reduce_only is True
    assert delta.close_position is True


# One-way short-side scenarios


async def test_open_short_from_flat() -> None:
    """No position → open short (sell 0.5, reduce_only=False)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot()
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("-0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.SELL
    assert delta.quantity == Decimal("0.5")
    assert delta.reduce_only is False


async def test_increase_short() -> None:
    """-0.3 short → -0.5 short = sell 0.2 (reduce_only=False)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("-0.3")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("-0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.SELL
    assert delta.quantity == Decimal("0.2")
    assert delta.reduce_only is False


async def test_reduce_short() -> None:
    """-0.5 short → -0.2 short = buy 0.3 (reduce_only=True)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("-0.5")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("-0.2"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.BUY
    assert delta.quantity == Decimal("0.3")
    assert delta.reduce_only is True
    assert delta.close_position is False


async def test_close_short() -> None:
    """-0.3 short → 0 = buy 0.3 (reduce_only=True, close_position=True)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("-0.3")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    delta = plan.deltas[0]
    assert delta.side == OrderSide.BUY
    assert delta.quantity == Decimal("0.3")
    assert delta.reduce_only is True
    assert delta.close_position is True


# Flip sequencing


async def test_flip_short_to_long_sequencing() -> None:
    """-0.3 short → 0.5 long: close short (seq=0) then open long (seq=1)."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot(positions=(_long_leg(Decimal("-0.3")),))
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.NET,
            target_qty=Decimal("0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 2
    assert plan.deltas[0].sequence == 0
    assert plan.deltas[0].reduce_only is True
    assert plan.deltas[0].close_position is True
    assert plan.deltas[0].side == OrderSide.BUY
    assert plan.deltas[0].quantity == Decimal("0.3")
    assert plan.deltas[1].sequence == 1
    assert plan.deltas[1].reduce_only is False
    assert plan.deltas[1].side == OrderSide.BUY
    assert plan.deltas[1].quantity == Decimal("0.5")


async def test_one_way_net_only_profile() -> None:
    """One-way mode uses PositionSide.NET for all operations."""
    planner = DefaultExecutionPlanner(position_mode=PositionMode.ONE_WAY)
    snapshot = _snapshot()
    targets = (
        TargetLegPosition(
            account="default",
            instrument_id=BTCUSDC,
            position_side=PositionSide.LONG,
            target_qty=Decimal("0.5"),
        ),
    )

    plan = await planner.plan(snapshot, targets)
    assert len(plan.deltas) == 1
    assert plan.deltas[0].position_side == PositionSide.NET

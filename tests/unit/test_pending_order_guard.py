"""Unit tests for the pending order guard in TickService."""

from decimal import Decimal

import pytest

from dojiwick.application.use_cases.run_tick import TickService
from dojiwick.domain.enums import (
    OrderSide,
    OrderType,
    PositionSide,
)
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from fixtures.fakes.pending_order_provider import FakePendingOrderProvider

_IID_BTC = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDC",
    base_asset="BTC",
    quote_asset="USDC",
    settle_asset="USDC",
)
_IID_ETH = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="ETHUSDC",
    base_asset="ETH",
    quote_asset="USDC",
    settle_asset="USDC",
)


def _delta(
    *,
    iid: InstrumentId = _IID_BTC,
    side: OrderSide = OrderSide.BUY,
    qty: Decimal = Decimal("0.5"),
    position_side: PositionSide = PositionSide.LONG,
    reduce_only: bool = False,
) -> LegDelta:
    return LegDelta(
        instrument_id=iid,
        target_index=0,
        position_side=position_side,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        reduce_only=reduce_only,
    )


def _plan(*deltas: LegDelta) -> ExecutionPlan:
    return ExecutionPlan(account="default", deltas=deltas)


async def _apply_guard(
    plan: ExecutionPlan,
    provider: FakePendingOrderProvider,
) -> ExecutionPlan:
    """Helper: call _apply_pending_order_guard directly via a minimal TickService stub."""
    # We only need pending_order_provider to be set on the service.
    # _apply_pending_order_guard is a standalone method that only uses self.pending_order_provider.
    svc = object.__new__(TickService)
    svc.pending_order_provider = provider  # type: ignore[attr-defined]
    return await svc._apply_pending_order_guard(plan)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_no_pending_unchanged() -> None:
    provider = FakePendingOrderProvider()
    plan = _plan(_delta(qty=Decimal("0.5")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 1
    assert result.deltas[0].quantity == Decimal("0.5")


@pytest.mark.asyncio
async def test_pending_partial_deduction() -> None:
    provider = FakePendingOrderProvider()
    provider.seed("BTCUSDC", PositionSide.LONG, Decimal("0.3"))
    plan = _plan(_delta(qty=Decimal("0.5")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 1
    assert result.deltas[0].quantity == Decimal("0.2")


@pytest.mark.asyncio
async def test_pending_covers_delta() -> None:
    provider = FakePendingOrderProvider()
    provider.seed("BTCUSDC", PositionSide.LONG, Decimal("0.5"))
    plan = _plan(_delta(qty=Decimal("0.3")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 0
    assert result.is_empty


@pytest.mark.asyncio
async def test_pending_different_symbol() -> None:
    provider = FakePendingOrderProvider()
    provider.seed("ETHUSDC", PositionSide.LONG, Decimal("0.5"))
    plan = _plan(_delta(iid=_IID_BTC, qty=Decimal("0.5")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 1
    assert result.deltas[0].quantity == Decimal("0.5")


@pytest.mark.asyncio
async def test_pending_different_side() -> None:
    provider = FakePendingOrderProvider()
    provider.seed("BTCUSDC", PositionSide.LONG, Decimal("0.5"))
    plan = _plan(_delta(position_side=PositionSide.SHORT, qty=Decimal("0.5")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 1
    assert result.deltas[0].quantity == Decimal("0.5")


@pytest.mark.asyncio
async def test_reduce_only_not_affected() -> None:
    provider = FakePendingOrderProvider()
    provider.seed("BTCUSDC", PositionSide.LONG, Decimal("0.5"))
    plan = _plan(_delta(reduce_only=True, side=OrderSide.SELL, qty=Decimal("0.5")))

    result = await _apply_guard(plan, provider)

    assert len(result.deltas) == 1
    assert result.deltas[0].quantity == Decimal("0.5")

"""Tests for SimulatedExecutionGateway."""

from decimal import Decimal

from dojiwick.domain.enums import ExecutionStatus, OrderSide, OrderType, PositionSide, SubmissionStatus
from dojiwick.domain.models.value_objects.cost_model import CostModel
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.infrastructure.exchange.simulated.execution import SimulatedExecutionGateway
from fixtures.fakes.clock import FixedClock


def _instrument(symbol: str = "BTCUSDC") -> InstrumentId:
    return InstrumentId(
        venue=VenueCode("binance"),
        product=ProductCode("usd_c"),
        symbol=symbol,
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )


def _plan(
    symbol: str = "BTCUSDC",
    side: OrderSide = OrderSide.BUY,
    quantity: Decimal = Decimal("0.01"),
) -> ExecutionPlan:
    return ExecutionPlan(
        account="default",
        deltas=(
            LegDelta(
                instrument_id=_instrument(symbol),
                target_index=0,
                position_side=PositionSide.NET,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                sequence=0,
            ),
        ),
    )


async def test_simulated_execution_gateway_fill() -> None:
    """Fill price includes slippage from CostModel."""
    cost = CostModel(slippage_bps=10.0)
    prices = {"BTCUSDC": Decimal("50000")}
    gateway = SimulatedExecutionGateway(cost_model=cost, prices=prices, clock=FixedClock())

    receipts = await gateway.execute_plan(_plan())

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt.status is ExecutionStatus.FILLED
    assert receipt.filled_quantity == Decimal("0.01")
    # BUY slippage: 50000 * (1 + 10/10000) = 50000 * 1.001 = 50050
    assert receipt.fill_price == Decimal("50000") * (1 + Decimal("10") / Decimal("10000"))


async def test_simulated_execution_gateway_sell_slippage() -> None:
    """SELL slippage reduces fill price."""
    cost = CostModel(slippage_bps=10.0)
    prices = {"BTCUSDC": Decimal("50000")}
    gateway = SimulatedExecutionGateway(cost_model=cost, prices=prices, clock=FixedClock())

    receipts = await gateway.execute_plan(_plan(side=OrderSide.SELL))

    receipt = receipts[0]
    assert receipt.status is ExecutionStatus.FILLED
    # SELL slippage: 50000 * (1 - 10/10000) = 50000 * 0.999 = 49950
    assert receipt.fill_price == Decimal("50000") * (1 - Decimal("10") / Decimal("10000"))


async def test_simulated_execution_gateway_unknown_symbol() -> None:
    """Unknown symbol produces REJECTED receipt."""
    cost = CostModel()
    prices = {"ETHUSDC": Decimal("3000")}
    gateway = SimulatedExecutionGateway(cost_model=cost, prices=prices, clock=FixedClock())

    receipts = await gateway.execute_plan(_plan(symbol="BTCUSDC"))

    assert len(receipts) == 1
    assert receipts[0].status is ExecutionStatus.REJECTED


async def test_simulated_execution_gateway_cancel_noop() -> None:
    """Cancel is a no-op returning ACCEPTED."""
    cost = CostModel()
    gateway = SimulatedExecutionGateway(cost_model=cost, prices={}, clock=FixedClock())

    ack = await gateway.cancel_order("BTCUSDC", "order_123")

    assert ack.status is SubmissionStatus.ACCEPTED

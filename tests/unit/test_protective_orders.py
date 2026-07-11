"""Protective-order reconciler lifecycle tests."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from dojiwick.application.services.protective_orders import ProtectiveOrderService
from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder
from dojiwick.domain.enums import OrderKind, OrderSide, OrderStatus, PositionSide
from dojiwick.domain.hashing import compute_protective_client_order_id
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter, InstrumentInfo
from dojiwick.domain.models.value_objects.position_leg import PositionLeg
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from fixtures.factories.infrastructure import default_settings
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.execution import DryRunGateway
from fixtures.fakes.instrument_repository import FakeInstrumentRepo
from fixtures.fakes.open_order import FakeOpenOrderAdapter
from fixtures.fakes.order_request_repository import FakeOrderRequestRepo
from fixtures.fakes.position_exit_state_repository import FakePositionExitStateRepository
from fixtures.fakes.position_leg_repository import FakePositionLegRepo

_IID = InstrumentId(
    venue=BINANCE_VENUE,
    product=BINANCE_USD_C,
    symbol="BTCUSDC",
    base_asset="BTC",
    quote_asset="USDC",
    settle_asset="USDC",
)
_INFO = InstrumentInfo(
    instrument_id=_IID,
    status="TRADING",
    filters=InstrumentFilter(
        tick_size=Decimal("0.1"),
        min_qty=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("5"),
    ),
    price_precision=1,
    quantity_precision=3,
    base_asset_precision=8,
    quote_asset_precision=8,
)


class _Env:
    def __init__(self) -> None:
        self.instrument_repo = FakeInstrumentRepo()
        db_id = self.instrument_repo.seed(BINANCE_VENUE, BINANCE_USD_C, "BTCUSDC", db_id=7)
        self.instrument_repo._instruments[db_id] = _INFO  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        self.leg_repo = FakePositionLegRepo()
        self.exit_repo = FakePositionExitStateRepository()
        self.open_orders = FakeOpenOrderAdapter()
        self.gateway = DryRunGateway()
        self.request_repo = FakeOrderRequestRepo()
        self.service = ProtectiveOrderService(
            settings=default_settings(),
            execution_gateway=self.gateway,
            open_order_port=self.open_orders,
            exchange_metadata=_StaticMetadata(),
            order_request_repo=self.request_repo,
            position_leg_repo=self.leg_repo,
            exit_state_repo=self.exit_repo,
            instrument_repo=self.instrument_repo,
            clock=FixedClock(datetime(2026, 1, 1, tzinfo=UTC)),
            account="default",
        )

    async def open_leg(self, qty: str = "0.010") -> int:
        leg = PositionLeg(
            account="default",
            instrument_id=7,
            position_side=PositionSide.LONG,
            quantity=Decimal(qty),
            entry_price=Decimal("50000"),
            opened_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        return await self.leg_repo.insert_leg(leg)


class _StaticMetadata:
    async def get_instrument(self, instrument_id: InstrumentId) -> InstrumentInfo:
        return _INFO

    async def list_instruments(self, venue: Any, product: Any) -> tuple[InstrumentInfo, ...]:
        return (_INFO,)

    async def get_capabilities(self, venue: Any, product: Any) -> Any:
        raise NotImplementedError


async def test_entry_registration_places_stop_and_tp() -> None:
    env = _Env()
    leg_id = await env.open_leg()
    await env.service.register_entry(
        position_leg_id=leg_id,
        is_long=True,
        entry_price=50_000.0,
        stop_price=49_000.0,
        take_profit_price=52_000.0,
        trailing_activation_price=0.0,
        trailing_distance=0.0,
        breakeven_price=0.0,
        max_hold_bars=0,
        tp1_price=0.0,
        tp1_fraction=0.0,
    )

    await env.service.sync()

    kinds = [r.order_kind for r in env.request_repo.requests]
    assert kinds.count(OrderKind.PROTECTIVE_STOP) == 1
    assert kinds.count(OrderKind.PROTECTIVE_TP) == 1
    assert all(r.reduce_only for r in env.request_repo.requests)
    assert all(r.position_leg_id == leg_id for r in env.request_repo.requests)


async def test_sync_is_idempotent_when_orders_rest() -> None:
    env = _Env()
    leg_id = await env.open_leg()
    await env.service.register_entry(
        position_leg_id=leg_id,
        is_long=True,
        entry_price=50_000.0,
        stop_price=49_000.0,
        take_profit_price=52_000.0,
        trailing_activation_price=0.0,
        trailing_distance=0.0,
        breakeven_price=0.0,
        max_hold_bars=0,
        tp1_price=0.0,
        tp1_fraction=0.0,
    )
    state = await env.exit_repo.get(leg_id)
    assert state is not None
    env.open_orders.seed(
        "BTCUSDC",
        [
            ExchangeOpenOrder(
                exchange_order_id="e1",
                client_order_id=compute_protective_client_order_id(
                    leg_id, OrderKind.PROTECTIVE_STOP.value, state.revision
                ),
                symbol="BTCUSDC",
                side=OrderSide.SELL,
                position_side=PositionSide.LONG,
                status=OrderStatus.NEW,
                original_quantity=Decimal("0.010"),
            ),
            ExchangeOpenOrder(
                exchange_order_id="e2",
                client_order_id=compute_protective_client_order_id(
                    leg_id, OrderKind.PROTECTIVE_TP.value, state.revision
                ),
                symbol="BTCUSDC",
                side=OrderSide.SELL,
                position_side=PositionSide.LONG,
                status=OrderStatus.NEW,
                original_quantity=Decimal("0.010"),
            ),
        ],
    )

    await env.service.sync()

    assert env.request_repo.requests == []
    assert env.open_orders.cancel_calls == []


async def test_orphan_protective_orders_cancelled() -> None:
    """Resting protective orders for a closed leg are retired by sync."""
    env = _Env()
    env.open_orders.seed(
        "BTCUSDC",
        [
            ExchangeOpenOrder(
                exchange_order_id="stale1",
                client_order_id="dw_p99_0_abc",
                symbol="BTCUSDC",
                side=OrderSide.SELL,
                position_side=PositionSide.LONG,
                status=OrderStatus.NEW,
                original_quantity=Decimal("0.010"),
            )
        ],
    )
    leg_id = await env.open_leg()
    await env.service.register_entry(
        position_leg_id=leg_id,
        is_long=True,
        entry_price=50_000.0,
        stop_price=49_000.0,
        take_profit_price=52_000.0,
        trailing_activation_price=0.0,
        trailing_distance=0.0,
        breakeven_price=0.0,
        max_hold_bars=0,
        tp1_price=0.0,
        tp1_fraction=0.0,
    )

    await env.service.sync()

    assert "BTCUSDC:stale1" in env.gateway.cancelled


async def test_trailing_amendment_bumps_revision() -> None:
    """A moved trailing stop yields a new client id — sync replaces the order."""
    env = _Env()
    leg_id = await env.open_leg()
    await env.service.register_entry(
        position_leg_id=leg_id,
        is_long=True,
        entry_price=50_000.0,
        stop_price=49_000.0,
        take_profit_price=60_000.0,
        trailing_activation_price=50_500.0,
        trailing_distance=300.0,
        breakeven_price=0.0,
        max_hold_bars=0,
        tp1_price=0.0,
        tp1_fraction=0.0,
    )
    before = await env.exit_repo.get(leg_id)
    assert before is not None
    rev_before = before.revision

    await env.service.update_trailing({"BTCUSDC": 51_000.0})

    after = await env.exit_repo.get(leg_id)
    assert after is not None
    assert after.stop_price > 49_000.0
    assert after.revision == rev_before + 1

"""Exchange-side protective exit orders — desired-state reconciler.

Every open position leg carries a ``PositionExitState`` (the live twin of the
backtest's exit fields). ``sync()`` derives the protective STOP/TP orders that
state implies, compares them with what is resting on the exchange, and
cancels/places the difference. The same idempotent pass runs at end-of-tick,
at startup, and periodically — trailing amendment, quantity drift after
partial fills, and crash recovery are all the one code path.

The exchange has no native OCO: when a protective order fills, the WS
consumer cancels the sibling; a race that leaves both resting is healed by
the next sync.
"""

import logging
from dataclasses import dataclass

from dojiwick.application.models.pipeline_settings import PipelineSettings
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.exchange_metadata import ExchangeMetadataPort
from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder, OpenOrderPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.contracts.repositories.order_request import OrderRequestRepositoryPort
from dojiwick.domain.contracts.repositories.position_exit_state import PositionExitStateRepositoryPort
from dojiwick.domain.contracts.repositories.position_leg import PositionLegRepositoryPort
from dojiwick.domain.enums import (
    OrderKind,
    OrderSide,
    OrderType,
    PositionSide,
    SubmissionStatus,
    TradeAction,
)
from dojiwick.domain.exit_rules import should_time_exit, update_trailing_stop
from dojiwick.domain.hashing import compute_protective_client_order_id
from dojiwick.domain.models.entities.position_exit_state import PositionExitState
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentInfo
from dojiwick.domain.models.value_objects.order_request import OrderRequest
from dojiwick.domain.models.value_objects.position_leg import PositionLeg
from dojiwick.domain.models.value_objects.protective_orders import ProtectiveOrderSpec
from dojiwick.domain.numerics import Price, Quantity, quantize_qty_to_step, round_price_to_tick, to_price, to_quantity

log = logging.getLogger(__name__)

_PROTECTIVE_PREFIX = "dw_p"


@dataclass(slots=True)
class ProtectiveOrderService:
    """Derives, places, amends, and retires exchange-side exit orders."""

    settings: PipelineSettings
    execution_gateway: ExecutionGatewayPort
    open_order_port: OpenOrderPort
    exchange_metadata: ExchangeMetadataPort
    order_request_repo: OrderRequestRepositoryPort
    position_leg_repo: PositionLegRepositoryPort
    exit_state_repo: PositionExitStateRepositoryPort
    instrument_repo: InstrumentRepositoryPort
    clock: ClockPort
    account: str

    async def register_entry(
        self,
        *,
        position_leg_id: int,
        is_long: bool,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        trailing_activation_price: float,
        trailing_distance: float,
        breakeven_price: float,
        max_hold_bars: int,
        tp1_price: float,
        tp1_fraction: float,
    ) -> None:
        """Persist exit state for a freshly opened/added leg (DB write only)."""
        state = PositionExitState(
            position_leg_id=position_leg_id,
            is_long=is_long,
            entry_price=entry_price,
            stop_price=stop_price,
            original_stop=stop_price,
            take_profit_price=take_profit_price,
            trailing_activation_price=trailing_activation_price,
            trailing_distance=trailing_distance,
            breakeven_price=breakeven_price,
            extreme_price=entry_price,
            max_hold_bars=max_hold_bars,
            tp1_price=tp1_price,
            tp1_fraction=tp1_fraction,
        )
        await self.exit_state_repo.upsert(state)

    async def update_trailing(self, prices_by_symbol: dict[str, float]) -> None:
        """Advance trailing/breakeven stops from the latest prices, fire time exits.

        Live has no intra-tick high/low — the last price approximates both,
        which only makes trailing conservative (it never overshoots a wick).
        """
        legs = {leg.id: leg for leg in await self.position_leg_repo.get_active_legs(self.account) if leg.id}
        for state in await self.exit_state_repo.list_active(self.account):
            leg = legs.get(state.position_leg_id)
            if leg is None:
                continue
            info = await self.instrument_repo.get_by_id(leg.instrument_id)
            if info is None:
                continue
            price = prices_by_symbol.get(info.instrument_id.symbol)
            if price is None or price <= 0:
                continue

            state.bars_held += 1
            update_trailing_stop(state, price, price, state.is_long)
            await self.exit_state_repo.upsert(state)

            if should_time_exit(state):
                await self._market_close_leg(leg, info)

    async def sync(self) -> None:
        """Reconcile resting protective orders with the desired set per leg."""
        legs = [leg for leg in await self.position_leg_repo.get_active_legs(self.account) if leg.id is not None]
        states = {s.position_leg_id: s for s in await self.exit_state_repo.list_active(self.account)}

        desired: dict[str, tuple[ProtectiveOrderSpec, PositionLeg, InstrumentInfo]] = {}
        symbols: set[str] = set()
        for leg in legs:
            info = await self.instrument_repo.get_by_id(leg.instrument_id)
            if info is None:
                log.error("no instrument metadata for leg %s — cannot protect", leg.id)
                continue
            symbols.add(info.instrument_id.symbol)
            state = states.get(leg.id or 0)
            if state is None:
                log.warning("open leg %s has no exit state — position is unprotected", leg.id)
                continue
            for spec in self._desired_specs(leg, state, info):
                desired[spec.client_order_id] = (spec, leg, info)

        # Orphan protective orders: legs gone but orders possibly resting
        for state_leg_id in states:
            if all(leg.id != state_leg_id for leg in legs):
                await self.exit_state_repo.delete(state_leg_id)

        resting: dict[str, tuple[str, ExchangeOpenOrder]] = {}
        for symbol in symbols:
            for order in await self.open_order_port.get_open_orders(symbol):
                if order.client_order_id.startswith(_PROTECTIVE_PREFIX):
                    resting[order.client_order_id] = (symbol, order)

        for client_id, (symbol, order) in resting.items():
            if client_id not in desired:
                await self._cancel_tolerant(symbol, order.exchange_order_id)

        for client_id, (spec, leg, info) in desired.items():
            if client_id in resting:
                continue
            await self._place_protective(spec, leg, info)

    async def release_for_symbols(self, symbols: set[str]) -> None:
        """Cancel resting protective orders before reduce/flip legs execute.

        A reduce-only market close racing its own protective stop can double
        close; freeing the protection first removes the race.
        """
        for symbol in symbols:
            for order in await self.open_order_port.get_open_orders(symbol):
                if order.client_order_id.startswith(_PROTECTIVE_PREFIX):
                    await self._cancel_tolerant(symbol, order.exchange_order_id)

    async def on_leg_closed(self, position_leg_id: int, symbol: str) -> None:
        """Cancel the surviving sibling and retire exit state after a leg closes."""
        for order in await self.open_order_port.get_open_orders(symbol):
            if order.client_order_id.startswith(f"{_PROTECTIVE_PREFIX}{position_leg_id}_"):
                await self._cancel_tolerant(symbol, order.exchange_order_id)
        await self.exit_state_repo.delete(position_leg_id)

    def _desired_specs(
        self,
        leg: PositionLeg,
        state: PositionExitState,
        info: InstrumentInfo,
    ) -> tuple[ProtectiveOrderSpec, ...]:
        leg_id = leg.id
        assert leg_id is not None
        filters = info.filters
        exit_side = OrderSide.SELL if state.is_long else OrderSide.BUY
        entry = to_price(str(state.entry_price))

        def _spec(
            kind: OrderKind, order_type: OrderType, trigger_price: Price, quantity: Quantity
        ) -> ProtectiveOrderSpec:
            return ProtectiveOrderSpec(
                kind=kind,
                position_leg_id=leg_id,
                instrument_id=info.instrument_id,
                position_side=leg.position_side,
                side=exit_side,
                order_type=order_type,
                trigger_price=trigger_price,
                quantity=quantity,
                working_type=self.settings.exchange.protective_working_type,
                price_protect=self.settings.exchange.protective_price_protect,
                client_order_id=compute_protective_client_order_id(leg_id, kind.value, state.revision),
            )

        specs: list[ProtectiveOrderSpec] = []

        stop_qty = quantize_qty_to_step(leg.quantity, filters.step_size)
        if stop_qty > 0:
            specs.append(
                _spec(
                    OrderKind.PROTECTIVE_STOP,
                    OrderType.STOP_MARKET,
                    round_price_to_tick(to_price(str(state.stop_price)), filters.tick_size, away_from=entry),
                    stop_qty,
                )
            )

        tp1_qty = to_quantity(0)
        if state.tp1_price > 0 and not state.tp1_filled and state.tp1_fraction > 0:
            tp1_qty = quantize_qty_to_step(leg.quantity * to_quantity(str(state.tp1_fraction)), filters.step_size)
            if tp1_qty > 0:
                specs.append(
                    _spec(
                        OrderKind.PROTECTIVE_TP1,
                        OrderType.TAKE_PROFIT_MARKET,
                        round_price_to_tick(to_price(str(state.tp1_price)), filters.tick_size),
                        tp1_qty,
                    )
                )

        tp_qty = quantize_qty_to_step(leg.quantity - tp1_qty, filters.step_size)
        if tp_qty > 0:
            specs.append(
                _spec(
                    OrderKind.PROTECTIVE_TP,
                    OrderType.TAKE_PROFIT_MARKET,
                    round_price_to_tick(to_price(str(state.take_profit_price)), filters.tick_size),
                    tp_qty,
                )
            )

        return tuple(specs)

    async def _place_protective(self, spec: ProtectiveOrderSpec, leg: PositionLeg, info: InstrumentInfo) -> None:
        # Pre-persist so the WS consumer can resolve the fill the moment the
        # exchange echoes it — protective fills arrive ONLY via WS
        request = OrderRequest(
            client_order_id=spec.client_order_id,
            instrument_id=leg.instrument_id,
            account=self.account,
            venue=str(spec.instrument_id.venue),
            product=str(spec.instrument_id.product),
            side=spec.side,
            order_type=spec.order_type,
            quantity=spec.quantity,
            price=spec.trigger_price,
            position_side=spec.position_side,
            reduce_only=True,
            working_type=spec.working_type,
            price_protect=spec.price_protect,
            order_kind=spec.kind,
            position_leg_id=spec.position_leg_id,
        )
        await self.order_request_repo.insert_request(request)

        action = TradeAction.SHORT if spec.side is OrderSide.SELL else TradeAction.BUY
        ack = await self.execution_gateway.place_order(
            spec.instrument_id.symbol,
            action,
            spec.order_type,
            spec.trigger_price,
            spec.quantity,
            instrument_id=spec.instrument_id,
            client_order_id=spec.client_order_id,
            position_side=spec.position_side,
            reduce_only=True,
            working_type=spec.working_type,
            price_protect=spec.price_protect,
        )
        if ack.status is not SubmissionStatus.ACCEPTED:
            log.error(
                "protective %s for leg %s rejected: %s — position unprotected until next sync",
                spec.kind.value,
                spec.position_leg_id,
                ack.reason,
            )

    async def _market_close_leg(self, leg: PositionLeg, info: InstrumentInfo) -> None:
        """Time exit: close the leg at market with a reduce-only order."""
        assert leg.id is not None
        symbol = info.instrument_id.symbol
        await self.on_leg_closed(leg.id, symbol)
        client_id = compute_protective_client_order_id(leg.id, OrderKind.EXIT.value, self.clock.epoch_ms())
        request = OrderRequest(
            client_order_id=client_id,
            instrument_id=leg.instrument_id,
            account=self.account,
            venue=str(info.instrument_id.venue),
            product=str(info.instrument_id.product),
            side=OrderSide.SELL if leg.position_side is not PositionSide.SHORT else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=leg.quantity,
            position_side=leg.position_side,
            reduce_only=True,
            order_kind=OrderKind.EXIT,
            position_leg_id=leg.id,
        )
        await self.order_request_repo.insert_request(request)
        action = TradeAction.SHORT if request.side is OrderSide.SELL else TradeAction.BUY
        ack = await self.execution_gateway.place_order(
            symbol,
            action,
            OrderType.MARKET,
            leg.entry_price,
            leg.quantity,
            instrument_id=info.instrument_id,
            client_order_id=client_id,
            position_side=leg.position_side,
            reduce_only=True,
        )
        if ack.status is not SubmissionStatus.ACCEPTED:
            log.error("time-exit close for leg %s rejected: %s", leg.id, ack.reason)

    async def _cancel_tolerant(self, symbol: str, exchange_order_id: str) -> None:
        """Cancel, treating already-gone orders as success (the sibling fired)."""
        ack = await self.execution_gateway.cancel_order(symbol, exchange_order_id)
        if ack.status is SubmissionStatus.ERROR and ack.reason != "order_not_found":
            log.warning("cancel of %s on %s failed: %s", exchange_order_id, symbol, ack.reason)

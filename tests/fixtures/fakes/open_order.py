"""Fake open order adapter for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.contracts.gateways.open_order import ExchangeOpenOrder


@dataclass(slots=True)
class FakeOpenOrderAdapter:
    """In-memory open order adapter for test assertions."""

    _orders: dict[str, list[ExchangeOpenOrder]] = field(default_factory=dict)
    cancel_calls: list[str] = field(default_factory=list)

    def seed(self, symbol: str, orders: list[ExchangeOpenOrder]) -> None:
        self._orders[symbol] = list(orders)

    async def get_open_orders(self, symbol: str) -> tuple[ExchangeOpenOrder, ...]:
        return tuple(self._orders.get(symbol, []))

    async def cancel_all_open_orders(self, symbol: str) -> None:
        self.cancel_calls.append(symbol)
        self._orders[symbol] = []

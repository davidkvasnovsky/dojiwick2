"""Startup order cleanup value objects."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.enums import OrderStatus
from dojiwick.domain.numerics import Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class StaleOrderRecord:
    """Record of a stale order found and cancelled during startup."""

    symbol: str
    exchange_order_id: str
    client_order_id: str
    status: OrderStatus
    filled_quantity: Quantity = Decimal(0)


@dataclass(slots=True, frozen=True)
class StartupCleanupResult:
    """Result of startup order cleanup across all symbols."""

    cancelled: tuple[StaleOrderRecord, ...]
    errors: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return len(self.cancelled) == 0 and len(self.errors) == 0

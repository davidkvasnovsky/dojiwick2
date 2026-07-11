"""Value objects for exchange-side protective exit orders."""

from dataclasses import dataclass

from dojiwick.domain.enums import OrderKind, OrderSide, OrderType, PositionSide, WorkingType
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.numerics import Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class ProtectiveOrderSpec:
    """One desired protective order for an open position leg."""

    kind: OrderKind
    position_leg_id: int
    instrument_id: InstrumentId
    position_side: PositionSide
    side: OrderSide
    order_type: OrderType
    trigger_price: Price
    quantity: Quantity
    working_type: WorkingType
    price_protect: bool
    client_order_id: str

    def __post_init__(self) -> None:
        if self.trigger_price <= 0:
            raise ValueError("trigger_price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

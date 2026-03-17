"""Exchange-oriented domain identity and target types."""

from dataclasses import dataclass

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.numerics import Money, Quantity


@dataclass(slots=True, frozen=True)
class InstrumentId:
    """Uniquely identifies a tradeable instrument on a venue."""

    venue: VenueCode
    product: ProductCode
    symbol: str
    base_asset: str
    quote_asset: str
    settle_asset: str

    def __post_init__(self) -> None:
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if not self.product:
            raise ValueError("product must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.base_asset:
            raise ValueError("base_asset must not be empty")
        if not self.quote_asset:
            raise ValueError("quote_asset must not be empty")
        if not self.settle_asset:
            raise ValueError("settle_asset must not be empty")


@dataclass(slots=True, frozen=True)
class PositionLegKey:
    """Natural key for a single position leg (account + instrument + side)."""

    account: str
    instrument_id: InstrumentId
    position_side: PositionSide

    def __post_init__(self) -> None:
        if not self.account:
            raise ValueError("account must not be empty")


@dataclass(slots=True, frozen=True)
class TargetLegPosition:
    """Desired position state for a single leg — input to the execution planner."""

    account: str
    instrument_id: InstrumentId
    position_side: PositionSide
    target_notional: Money | None = None
    target_qty: Quantity | None = None

    def __post_init__(self) -> None:
        if not self.account:
            raise ValueError("account must not be empty")
        if self.target_notional is None and self.target_qty is None:
            raise ValueError("at least one of target_notional or target_qty must be provided")

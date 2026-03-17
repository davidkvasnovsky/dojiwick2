"""Instrument metadata value objects for exchange adapter boundaries."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class InstrumentFilter:
    """Price / quantity / notional filter rules for an instrument."""

    min_price: Price = Decimal(0)
    max_price: Price | None = None
    tick_size: Price = Decimal(0)
    min_qty: Quantity = Decimal(0)
    max_qty: Quantity | None = None
    step_size: Quantity = Decimal(0)
    min_notional: Money = Decimal(0)


@dataclass(slots=True, frozen=True, kw_only=True)
class InstrumentInfo:
    """Full metadata for a single instrument — returned by ExchangeMetadataPort."""

    instrument_id: InstrumentId
    status: str
    filters: InstrumentFilter
    price_precision: int
    quantity_precision: int
    base_asset_precision: int
    quote_asset_precision: int
    contract_size: Quantity = Decimal(1)
    margin_asset: str = ""


@dataclass(slots=True, frozen=True, kw_only=True)
class ExchangeCapabilities:
    """Venue + product level capabilities."""

    venue: VenueCode
    product: ProductCode
    supports_hedge_mode: bool
    supports_stop_market: bool
    supports_take_profit_market: bool
    max_open_orders: int
    supported_time_in_force: tuple[str, ...]

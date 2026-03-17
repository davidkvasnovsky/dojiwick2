"""Binance exchange metadata adapter — instruments, filters, and capabilities."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.errors import ExchangeError
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import (
    ExchangeCapabilities,
    InstrumentFilter,
    InstrumentInfo,
)

from .boundary import build_instrument_id
from .http_client import BinanceHttpClient
from .response_types import ExchangeInfoResponse, FilterEntry, SymbolInfo


@dataclass(slots=True)
class BinanceExchangeMetadataProvider:
    """Fetches instrument metadata and capabilities from Binance Futures."""

    client: BinanceHttpClient
    _exchange_info: ExchangeInfoResponse | None = field(default=None, init=False, repr=False)

    async def _fetch_exchange_info(self) -> ExchangeInfoResponse:
        """GET /fapi/v1/exchangeInfo with caching."""
        if self._exchange_info is None:
            self._exchange_info = cast(
                ExchangeInfoResponse,
                await self.client.request("GET", "/fapi/v1/exchangeInfo"),
            )
        return self._exchange_info

    async def list_instruments(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> tuple[InstrumentInfo, ...]:
        """Return metadata for all TRADING instruments."""
        from dojiwick.infrastructure.exchange.binance.constants import BINANCE_VENUE

        if venue != BINANCE_VENUE:
            raise ExchangeError(f"BinanceExchangeMetadataProvider only supports venue={BINANCE_VENUE}, got {venue}")
        # product filtering not needed: /fapi/v1/exchangeInfo returns all linear futures
        info = await self._fetch_exchange_info()

        instruments: list[InstrumentInfo] = []
        for sym in info["symbols"]:
            if sym.get("status") != "TRADING":
                continue
            instruments.append(_parse_instrument(sym))
        return tuple(instruments)

    async def get_instrument(self, instrument_id: InstrumentId) -> InstrumentInfo:
        """Return metadata for a single instrument by InstrumentId."""
        instruments = await self.list_instruments(instrument_id.venue, instrument_id.product)
        for inst in instruments:
            if inst.instrument_id.symbol == instrument_id.symbol:
                return inst
        raise ExchangeError(f"instrument not found: {instrument_id.symbol}")

    async def get_capabilities(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> ExchangeCapabilities:
        """Return hardcoded Binance USD-C Futures capabilities."""
        return ExchangeCapabilities(
            venue=venue,
            product=product,
            supports_hedge_mode=True,
            supports_stop_market=True,
            supports_take_profit_market=True,
            max_open_orders=200,
            supported_time_in_force=("GTC", "IOC", "FOK", "GTX"),
        )


def _parse_instrument(sym: SymbolInfo) -> InstrumentInfo:
    """Parse a single symbol dict from exchangeInfo into InstrumentInfo."""
    symbol = sym["symbol"]
    base_asset = sym["baseAsset"]
    quote_asset = sym["quoteAsset"]
    margin_asset = sym["marginAsset"]
    settle_asset = sym.get("settlAsset", "") or quote_asset

    filters = _parse_filters(sym["filters"])

    return InstrumentInfo(
        instrument_id=build_instrument_id(symbol, base_asset, quote_asset, settle_asset),
        status=sym["status"],
        filters=filters,
        price_precision=sym.get("pricePrecision", 8),
        quantity_precision=sym.get("quantityPrecision", 8),
        base_asset_precision=sym.get("baseAssetPrecision", 8),
        quote_asset_precision=sym.get("quotePrecision", 8),
        margin_asset=margin_asset,
    )


def _parse_filters(raw_filters: list[FilterEntry]) -> InstrumentFilter:
    """Parse the filters array from a symbol entry."""
    min_price = Decimal(0)
    max_price: Decimal | None = None
    tick_size = Decimal(0)
    min_qty = Decimal(0)
    max_qty: Decimal | None = None
    step_size = Decimal(0)
    min_notional = Decimal(0)

    for f in raw_filters:
        filter_type = f.get("filterType")
        if filter_type == "PRICE_FILTER":
            min_price = Decimal(f.get("minPrice", "0"))
            raw_max = f.get("maxPrice", "0")
            max_price = Decimal(raw_max) if raw_max != "0" else None
            tick_size = Decimal(f.get("tickSize", "0"))
        elif filter_type == "LOT_SIZE":
            min_qty = Decimal(f.get("minQty", "0"))
            raw_max_qty = f.get("maxQty", "0")
            max_qty = Decimal(raw_max_qty) if raw_max_qty != "0" else None
            step_size = Decimal(f.get("stepSize", "0"))
        elif filter_type == "MIN_NOTIONAL":
            min_notional = Decimal(f.get("notional", "0"))

    return InstrumentFilter(
        min_price=min_price,
        max_price=max_price,
        tick_size=tick_size,
        min_qty=min_qty,
        max_qty=max_qty,
        step_size=step_size,
        min_notional=min_notional,
    )

"""Exchange metadata port test doubles."""

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import ExchangeCapabilities, InstrumentInfo


class FakeExchangeMetadata:
    """In-memory fake for ExchangeMetadataPort — returns configurable instrument metadata."""

    def __init__(
        self,
        instruments: dict[InstrumentId, InstrumentInfo] | None = None,
        capabilities: dict[tuple[VenueCode, ProductCode], ExchangeCapabilities] | None = None,
    ) -> None:
        self._instruments: dict[InstrumentId, InstrumentInfo] = instruments or {}
        self._capabilities: dict[tuple[VenueCode, ProductCode], ExchangeCapabilities] = capabilities or {}

    def add_instrument(self, info: InstrumentInfo) -> None:
        """Test helper: register an instrument."""
        self._instruments[info.instrument_id] = info

    def set_capabilities(self, caps: ExchangeCapabilities) -> None:
        """Test helper: register capabilities for a venue+product."""
        self._capabilities[(caps.venue, caps.product)] = caps

    async def get_instrument(self, instrument_id: InstrumentId) -> InstrumentInfo:
        if instrument_id not in self._instruments:
            raise KeyError(f"instrument not found: {instrument_id}")
        return self._instruments[instrument_id]

    async def list_instruments(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> tuple[InstrumentInfo, ...]:
        return tuple(
            info
            for info in self._instruments.values()
            if info.instrument_id.venue == venue and info.instrument_id.product == product
        )

    async def get_capabilities(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> ExchangeCapabilities:
        key = (venue, product)
        if key not in self._capabilities:
            raise KeyError(f"capabilities not found for {venue}/{product}")
        return self._capabilities[key]

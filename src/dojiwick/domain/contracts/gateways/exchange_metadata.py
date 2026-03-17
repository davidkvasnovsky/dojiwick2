"""Exchange metadata port — fetching instrument metadata, filters, precision, and capabilities."""

from typing import Protocol

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import ExchangeCapabilities, InstrumentInfo


class ExchangeMetadataPort(Protocol):
    """Fetches instrument metadata, filters, precision, and capability info."""

    async def get_instrument(self, instrument_id: InstrumentId) -> InstrumentInfo:
        """Return metadata for a single instrument."""
        ...

    async def list_instruments(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> tuple[InstrumentInfo, ...]:
        """Return metadata for all instruments matching the venue/product filter."""
        ...

    async def get_capabilities(
        self,
        venue: VenueCode,
        product: ProductCode,
    ) -> ExchangeCapabilities:
        """Return venue+product level capabilities (hedge mode support, order types, etc.)."""
        ...

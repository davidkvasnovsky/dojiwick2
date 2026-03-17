"""Instrument repository protocol."""

from typing import Protocol

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentInfo


class InstrumentRepositoryPort(Protocol):
    """Exchange instrument metadata persistence."""

    async def upsert_instrument(self, info: InstrumentInfo) -> int:
        """Insert or update an instrument and its filters, returning the DB id."""
        ...

    async def get_by_symbol(self, venue: VenueCode, product: ProductCode, symbol: str) -> InstrumentInfo | None:
        """Return instrument info by (venue, product, symbol), or None."""
        ...

    async def get_by_id(self, instrument_id: int) -> InstrumentInfo | None:
        """Return instrument info by DB id, or None."""
        ...

    async def resolve_id(self, venue: VenueCode, product: ProductCode, symbol: str) -> int | None:
        """Return the DB integer id for an instrument, or None."""
        ...

    async def list_instruments(self, venue: VenueCode, product: ProductCode) -> tuple[InstrumentInfo, ...]:
        """Return all instruments for a venue and product."""
        ...

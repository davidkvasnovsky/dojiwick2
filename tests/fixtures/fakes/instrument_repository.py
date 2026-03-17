"""Fake instrument repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentInfo


@dataclass(slots=True)
class FakeInstrumentRepo:
    """In-memory instrument repository with symbol→int mapping for resolve_id."""

    _id_map: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _instruments: dict[int, InstrumentInfo] = field(default_factory=dict)
    _next_id: int = 1

    def seed(self, venue: VenueCode, product: ProductCode, symbol: str, db_id: int | None = None) -> int:
        """Pre-seed a symbol→id mapping. Returns the assigned id."""
        key = (venue, product, symbol)
        if key in self._id_map:
            return self._id_map[key]
        assigned = db_id if db_id is not None else self._next_id
        self._id_map[key] = assigned
        if db_id is None:
            self._next_id += 1
        return assigned

    async def resolve_id(self, venue: VenueCode, product: ProductCode, symbol: str) -> int | None:
        return self._id_map.get((venue, product, symbol))

    async def upsert_instrument(self, info: InstrumentInfo) -> int:
        iid = info.instrument_id
        key = (iid.venue, iid.product, iid.symbol)
        if key in self._id_map:
            db_id = self._id_map[key]
        else:
            db_id = self._next_id
            self._next_id += 1
            self._id_map[key] = db_id
        self._instruments[db_id] = info
        return db_id

    async def get_by_symbol(self, venue: VenueCode, product: ProductCode, symbol: str) -> InstrumentInfo | None:
        db_id = self._id_map.get((venue, product, symbol))
        if db_id is None:
            return None
        return self._instruments.get(db_id)

    async def get_by_id(self, instrument_id: int) -> InstrumentInfo | None:
        return self._instruments.get(instrument_id)

    async def list_instruments(self, venue: VenueCode, product: ProductCode) -> tuple[InstrumentInfo, ...]:
        return tuple(
            info
            for info in self._instruments.values()
            if info.instrument_id.venue == venue and info.instrument_id.product == product
        )

"""PostgreSQL instrument repository."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter, InstrumentInfo

from dojiwick.infrastructure.postgres.connection import DbConnection

_UPSERT_INSTRUMENT_SQL = """
INSERT INTO instruments (
    venue, product, symbol, base_asset, quote_asset, settle_asset,
    status, price_precision, quantity_precision,
    base_asset_precision, quote_asset_precision,
    contract_size, margin_asset, metadata, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT ON CONSTRAINT instruments_unique_symbol
DO UPDATE SET
    status = EXCLUDED.status,
    price_precision = EXCLUDED.price_precision,
    quantity_precision = EXCLUDED.quantity_precision,
    base_asset_precision = EXCLUDED.base_asset_precision,
    quote_asset_precision = EXCLUDED.quote_asset_precision,
    contract_size = EXCLUDED.contract_size,
    margin_asset = EXCLUDED.margin_asset,
    metadata = EXCLUDED.metadata,
    updated_at = now()
RETURNING id
"""

_DELETE_FILTERS_SQL = """
DELETE FROM instrument_filters WHERE instrument_id = %s
"""

_INSERT_FILTER_SQL = """
INSERT INTO instrument_filters (
    instrument_id, min_price, max_price, tick_size,
    min_qty, max_qty, step_size, min_notional
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_BY_SYMBOL_SQL = """
SELECT i.id, i.venue, i.product, i.symbol, i.base_asset, i.quote_asset, i.settle_asset,
       i.status, i.price_precision, i.quantity_precision,
       i.base_asset_precision, i.quote_asset_precision,
       i.contract_size, i.margin_asset,
       f.min_price, f.max_price, f.tick_size,
       f.min_qty, f.max_qty, f.step_size, f.min_notional
FROM instruments i
LEFT JOIN instrument_filters f ON f.instrument_id = i.id
WHERE i.venue = %s AND i.product = %s AND i.symbol = %s
"""

_SELECT_BY_ID_SQL = """
SELECT i.id, i.venue, i.product, i.symbol, i.base_asset, i.quote_asset, i.settle_asset,
       i.status, i.price_precision, i.quantity_precision,
       i.base_asset_precision, i.quote_asset_precision,
       i.contract_size, i.margin_asset,
       f.min_price, f.max_price, f.tick_size,
       f.min_qty, f.max_qty, f.step_size, f.min_notional
FROM instruments i
LEFT JOIN instrument_filters f ON f.instrument_id = i.id
WHERE i.id = %s
"""

_SELECT_BY_VENUE_PRODUCT_SQL = """
SELECT i.id, i.venue, i.product, i.symbol, i.base_asset, i.quote_asset, i.settle_asset,
       i.status, i.price_precision, i.quantity_precision,
       i.base_asset_precision, i.quote_asset_precision,
       i.contract_size, i.margin_asset,
       f.min_price, f.max_price, f.tick_size,
       f.min_qty, f.max_qty, f.step_size, f.min_notional
FROM instruments i
LEFT JOIN instrument_filters f ON f.instrument_id = i.id
WHERE i.venue = %s AND i.product = %s
ORDER BY i.symbol
"""


def _row_to_info(row: tuple[object, ...]) -> InstrumentInfo:
    """Map a joined instrument+filter row to InstrumentInfo."""
    (
        _db_id,
        venue,
        product,
        symbol,
        base_asset,
        quote_asset,
        settle_asset,
        status,
        price_precision,
        quantity_precision,
        base_asset_precision,
        quote_asset_precision,
        contract_size,
        margin_asset,
        min_price,
        max_price,
        tick_size,
        min_qty,
        max_qty,
        step_size,
        min_notional,
    ) = row
    instrument_id = InstrumentId(
        venue=VenueCode(str(venue)),
        product=ProductCode(str(product)),
        symbol=str(symbol),
        base_asset=str(base_asset),
        quote_asset=str(quote_asset),
        settle_asset=str(settle_asset),
    )
    filt = InstrumentFilter(
        min_price=Decimal(str(min_price)) if min_price is not None else Decimal(0),
        max_price=Decimal(str(max_price)) if max_price is not None else None,
        tick_size=Decimal(str(tick_size)) if tick_size is not None else Decimal(0),
        min_qty=Decimal(str(min_qty)) if min_qty is not None else Decimal(0),
        max_qty=Decimal(str(max_qty)) if max_qty is not None else None,
        step_size=Decimal(str(step_size)) if step_size is not None else Decimal(0),
        min_notional=Decimal(str(min_notional)) if min_notional is not None else Decimal(0),
    )
    return InstrumentInfo(
        instrument_id=instrument_id,
        status=str(status),
        filters=filt,
        price_precision=int(str(price_precision)),
        quantity_precision=int(str(quantity_precision)),
        base_asset_precision=int(str(base_asset_precision)),
        quote_asset_precision=int(str(quote_asset_precision)),
        contract_size=Decimal(str(contract_size)),
        margin_asset=str(margin_asset),
    )


@dataclass(slots=True)
class PgInstrumentRepository:
    """Persists instrument metadata into PostgreSQL."""

    connection: DbConnection

    async def upsert_instrument(self, info: InstrumentInfo) -> int:
        """Insert or update an instrument and its filters, returning the DB id."""
        iid = info.instrument_id
        row = (
            iid.venue,
            iid.product,
            iid.symbol,
            iid.base_asset,
            iid.quote_asset,
            iid.settle_asset,
            info.status,
            info.price_precision,
            info.quantity_precision,
            info.base_asset_precision,
            info.quote_asset_precision,
            info.contract_size,
            info.margin_asset,
            None,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_UPSERT_INSTRUMENT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert instrument: {exc}") from exc
        if result is None:
            raise AdapterError("upsert instrument returned no id")
        db_id = int(result[0])

        f = info.filters
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_DELETE_FILTERS_SQL, (db_id,))
                await cursor.execute(
                    _INSERT_FILTER_SQL,
                    (db_id, f.min_price, f.max_price, f.tick_size, f.min_qty, f.max_qty, f.step_size, f.min_notional),
                )
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert instrument filters: {exc}") from exc
        return db_id

    async def get_by_symbol(self, venue: VenueCode, product: ProductCode, symbol: str) -> InstrumentInfo | None:
        """Return instrument info by (venue, product, symbol), or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_SYMBOL_SQL, (venue, product, symbol))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get instrument: {exc}") from exc
        if row is None:
            return None
        return _row_to_info(row)

    async def get_by_id(self, instrument_id: int) -> InstrumentInfo | None:
        """Return instrument info by DB id, or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_ID_SQL, (instrument_id,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get instrument by id: {exc}") from exc
        if row is None:
            return None
        return _row_to_info(row)

    async def resolve_id(self, venue: VenueCode, product: ProductCode, symbol: str) -> int | None:
        """Return the DB integer id for an instrument, or None."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT id FROM instruments WHERE venue = %s AND product = %s AND symbol = %s",
                    (venue, product, symbol),
                )
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to resolve instrument id: {exc}") from exc
        if row is None:
            return None
        return int(row[0])

    async def list_instruments(self, venue: VenueCode, product: ProductCode) -> tuple[InstrumentInfo, ...]:
        """Return all instruments for a venue and product."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_VENUE_PRODUCT_SQL, (venue, product))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to list instruments: {exc}") from exc
        return tuple(_row_to_info(r) for r in rows)

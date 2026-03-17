"""PostgreSQL candle repository."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.type_aliases import CandleInterval

from dojiwick.infrastructure.postgres.connection import DbConnection

_UPSERT_SQL = """
INSERT INTO candles (venue, product, pair, timeframe, open_time, open, high, low, close, volume, quote_volume)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (venue, product, pair, timeframe, open_time)
DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
              close = EXCLUDED.close, volume = EXCLUDED.volume, quote_volume = EXCLUDED.quote_volume,
              updated_at = now()
"""

_SELECT_SQL = """
SELECT pair, timeframe, open_time, open, high, low, close, volume, quote_volume
FROM candles
WHERE pair = %s AND timeframe = %s AND open_time >= %s AND open_time <= %s
  AND venue = %s AND product = %s
ORDER BY open_time
"""


@dataclass(slots=True)
class PgCandleRepository:
    """Persists OHLCV data into PostgreSQL."""

    connection: DbConnection

    async def upsert_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        candles: tuple[Candle, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        if not venue or not product:
            raise AdapterError("upsert_candles requires non-empty venue and product")
        rows = [
            (
                venue,
                product,
                symbol,
                interval,
                c.open_time,
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.quote_volume,
            )
            for c in candles
        ]
        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_UPSERT_SQL, rows)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert candles: {exc}") from exc
        return len(rows)

    async def get_candles(
        self,
        symbol: str,
        interval: CandleInterval,
        start: datetime,
        end: datetime,
        *,
        venue: str,
        product: str,
    ) -> tuple[Candle, ...]:
        if not venue or not product:
            raise AdapterError("get_candles requires non-empty venue and product")
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (symbol, interval, start, end, venue, product))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get candles: {exc}") from exc
        return tuple(
            Candle(
                pair=row[0],
                interval=row[1],
                open_time=row[2],
                open=row[3],
                high=row[4],
                low=row[5],
                close=row[6],
                volume=row[7],
                quote_volume=row[8],
            )
            for row in rows
        )

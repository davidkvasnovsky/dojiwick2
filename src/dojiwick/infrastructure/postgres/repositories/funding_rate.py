"""PostgreSQL funding rate repository."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.funding_rate import FundingRate

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO funding_rates (venue, product, symbol, funding_time, funding_rate)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (venue, product, symbol, funding_time) DO NOTHING
"""

_SELECT_SQL = """
SELECT symbol, funding_time, funding_rate
FROM funding_rates
WHERE symbol = %s AND funding_time >= %s AND funding_time <= %s
  AND venue = %s AND product = %s
ORDER BY funding_time
"""


@dataclass(slots=True)
class PgFundingRateRepository:
    """Persists settled funding events into PostgreSQL (append-only)."""

    connection: DbConnection

    async def upsert_rates(
        self,
        symbol: str,
        rates: tuple[FundingRate, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        if not venue or not product:
            raise AdapterError("upsert_rates requires non-empty venue and product")
        rows = [(venue, product, symbol, r.funding_time, r.rate) for r in rates]
        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_INSERT_SQL, rows)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert funding rates: {exc}") from exc
        return len(rows)

    async def get_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        venue: str,
        product: str,
    ) -> tuple[FundingRate, ...]:
        if not venue or not product:
            raise AdapterError("get_rates requires non-empty venue and product")
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (symbol, start, end, venue, product))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get funding rates: {exc}") from exc
        return tuple(FundingRate(symbol=row[0], funding_time=row[1], rate=row[2]) for row in rows)

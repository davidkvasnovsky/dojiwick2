"""PostgreSQL performance snapshot repository."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO performance_snapshots (observed_at, equity_usd, unrealized_pnl_usd, realized_pnl_usd, open_positions, drawdown_pct)
VALUES (%s, %s, %s, %s, %s, %s)
"""

_SELECT_SQL = """
SELECT observed_at, equity_usd, unrealized_pnl_usd, realized_pnl_usd, open_positions, drawdown_pct
FROM performance_snapshots
WHERE observed_at >= %s AND observed_at <= %s
ORDER BY observed_at
"""


@dataclass(slots=True)
class PgPerformanceRepository:
    """Persists performance snapshots into PostgreSQL."""

    connection: DbConnection

    async def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        row = (
            snapshot.observed_at,
            snapshot.equity_usd,
            snapshot.unrealized_pnl_usd,
            snapshot.realized_pnl_usd,
            snapshot.open_positions,
            snapshot.drawdown_pct,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record snapshot: {exc}") from exc

    async def get_snapshots(self, start: datetime, end: datetime) -> tuple[PerformanceSnapshot, ...]:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (start, end))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get snapshots: {exc}") from exc
        return tuple(
            PerformanceSnapshot(
                observed_at=row[0],
                equity_usd=row[1],
                unrealized_pnl_usd=row[2],
                realized_pnl_usd=row[3],
                open_positions=row[4],
                drawdown_pct=row[5],
            )
            for row in rows
        )

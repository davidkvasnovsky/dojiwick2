"""PostgreSQL signal repository."""

import json
from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.signal import Signal

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO signals (pair, target_id, signal_type, priority, details, detected_at, decision_outcome_id, venue, product)
VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_SQL = """
SELECT pair, target_id, signal_type, priority, details, detected_at, decision_outcome_id
FROM signals
WHERE pair = %s AND detected_at >= %s AND detected_at <= %s
ORDER BY detected_at
"""


@dataclass(slots=True)
class PgSignalRepository:
    """Persists signals into PostgreSQL."""

    connection: DbConnection

    async def record_signal(self, signal: Signal, *, venue: str, product: str) -> int:
        if not venue or not product:
            raise AdapterError("record_signal requires non-empty venue and product")
        row = (
            signal.pair,
            signal.target_id,
            signal.signal_type,
            signal.priority,
            json.dumps(signal.details) if signal.details is not None else None,
            signal.detected_at,
            signal.decision_outcome_id,
            venue,
            product,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
                result = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record signal: {exc}") from exc
        if result is None:
            raise AdapterError("INSERT ... RETURNING id returned no row")
        return int(result[0])

    async def get_signals_for_tick(self, pair: str, start: datetime, end: datetime) -> tuple[Signal, ...]:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (pair, start, end))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get signals: {exc}") from exc
        return tuple(
            Signal(
                pair=row[0],
                target_id=row[1],
                signal_type=row[2],
                priority=row[3],
                details=row[4],
                detected_at=row[5],
                decision_outcome_id=row[6],
            )
            for row in rows
        )

"""PostgreSQL decision trace repository."""

import json
from dataclasses import dataclass

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace
from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO decision_traces (tick_id, step_name, step_seq, artifacts, step_hash, duration_us)
VALUES (%s, %s, %s, %s, %s, %s)
"""


@dataclass(slots=True)
class PgDecisionTraceRepository:
    """Persists decision traces into PostgreSQL."""

    connection: DbConnection

    async def insert_batch(self, traces: tuple[DecisionTrace, ...]) -> None:
        if not traces:
            return
        try:
            params = [
                (
                    t.tick_id,
                    t.step_name,
                    t.step_seq,
                    json.dumps(t.artifacts, default=str),
                    t.step_hash,
                    t.duration_us,
                )
                for t in traces
            ]
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_INSERT_SQL, params)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert decision traces: {exc}") from exc

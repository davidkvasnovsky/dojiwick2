"""PostgreSQL model cost repository."""

from dataclasses import dataclass

from dojiwick.domain.models.value_objects.model_cost import ModelCostRecord
from dojiwick.infrastructure.postgres.connection import DbConnection
from dojiwick.infrastructure.postgres.helpers import pg_execute, pg_execute_many

_INSERT_SQL = """
INSERT INTO model_costs (tick_id, model, input_tokens, output_tokens, cost_usd, purpose, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def _to_row(record: ModelCostRecord) -> tuple[str, str, int, int, str, str, str]:
    return (
        record.tick_id,
        record.model,
        record.input_tokens,
        record.output_tokens,
        str(record.cost_usd),
        record.purpose,
        record.created_at.isoformat(),
    )


@dataclass(slots=True)
class PgModelCostRepository:
    """Persists LLM cost records into PostgreSQL."""

    connection: DbConnection

    async def record_cost(self, record: ModelCostRecord) -> None:
        """Persist a model cost entry."""
        await pg_execute(self.connection, _INSERT_SQL, _to_row(record), error_msg="failed to record model cost")

    async def batch_record_costs(self, records: tuple[ModelCostRecord, ...]) -> None:
        """Persist multiple model cost entries in a single batch."""
        if not records:
            return
        rows = [_to_row(r) for r in records]
        await pg_execute_many(self.connection, _INSERT_SQL, rows, error_msg="failed to batch record model costs")

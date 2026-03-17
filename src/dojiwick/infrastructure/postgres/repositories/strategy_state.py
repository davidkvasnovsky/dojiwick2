"""PostgreSQL strategy state repository."""

import json
from dataclasses import dataclass
from typing import cast

from dojiwick.domain.errors import AdapterError

from dojiwick.infrastructure.postgres.connection import DbConnection

_UPSERT_SQL = """
INSERT INTO strategy_state (pair, active_strategy, variant, state_json, target_id, venue, product, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (pair, active_strategy, variant)
DO UPDATE SET
    state_json = EXCLUDED.state_json,
    target_id = EXCLUDED.target_id,
    venue = EXCLUDED.venue,
    product = EXCLUDED.product,
    updated_at = now()
"""

_SELECT_SQL = """
SELECT state_json FROM strategy_state
WHERE pair = %s AND active_strategy = %s AND variant = %s
"""


@dataclass(slots=True)
class PgStrategyStateRepository:
    """Persists per-pair strategy state into PostgreSQL."""

    connection: DbConnection

    async def get_state(self, pair: str, strategy_name: str, variant: str) -> dict[str, object] | None:
        """Return the stored state for a pair/strategy/variant, or None if absent."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_SQL, (pair, strategy_name, variant))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get strategy state: {exc}") from exc
        if row is None:
            return None
        state_json = row[0]
        if state_json is None:
            return None
        if isinstance(state_json, dict):
            return cast(dict[str, object], state_json)
        return cast(dict[str, object], json.loads(str(state_json)))

    async def upsert_state(
        self,
        pair: str,
        strategy_name: str,
        variant: str,
        state: dict[str, object],
        *,
        target_id: str,
        venue: str,
        product: str,
    ) -> None:
        """Insert or update the state for a pair/strategy/variant."""
        if not venue or not product:
            raise AdapterError("upsert_state requires non-empty venue and product")
        if not target_id:
            raise AdapterError("upsert_state requires non-empty target_id")
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    _UPSERT_SQL, (pair, strategy_name, variant, json.dumps(state), target_id, venue, product)
                )
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert strategy state: {exc}") from exc

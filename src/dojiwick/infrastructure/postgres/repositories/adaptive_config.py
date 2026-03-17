"""PostgreSQL adaptive config repository."""

import json
from dataclasses import dataclass
from typing import cast

from dojiwick.infrastructure.postgres.connection import DbConnection
from dojiwick.infrastructure.postgres.helpers import pg_fetch_all, pg_fetch_one

_SELECT_BY_IDX_SQL = """
SELECT config_idx, params_json
FROM adaptive_configs
WHERE config_idx = %s
"""

_SELECT_ALL_SQL = """
SELECT config_idx, params_json
FROM adaptive_configs
ORDER BY config_idx
"""


def _parse_params(raw: object) -> dict[str, object]:
    """Parse params_json from DB (JSONB may arrive as dict or str)."""
    if isinstance(raw, dict):
        return cast(dict[str, object], raw)
    loaded: dict[str, object] = json.loads(str(raw))
    return loaded


@dataclass(slots=True)
class PgAdaptiveConfigRepository:
    """Persists adaptive configuration into PostgreSQL."""

    connection: DbConnection

    async def get_config(self, config_idx: int) -> dict[str, object] | None:
        """Return a configuration by index, or None if absent."""
        row = await pg_fetch_one(
            self.connection, _SELECT_BY_IDX_SQL, (config_idx,), error_msg="failed to get adaptive config"
        )
        if row is None:
            return None
        return _parse_params(row[1])

    async def get_all_configs(self) -> tuple[tuple[int, dict[str, object]], ...]:
        """Return all configurations as (index, config) tuples."""
        rows = await pg_fetch_all(self.connection, _SELECT_ALL_SQL, error_msg="failed to get all adaptive configs")
        return tuple((int(str(row[0])), _parse_params(row[1])) for row in rows)

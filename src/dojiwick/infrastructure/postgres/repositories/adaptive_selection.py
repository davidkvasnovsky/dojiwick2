"""PostgreSQL adaptive selection repository."""

from dataclasses import dataclass

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveSelectionEvent
from dojiwick.infrastructure.postgres.connection import DbConnection
from dojiwick.infrastructure.postgres.helpers import parse_pg_datetime, pg_execute, pg_fetch_one

_INSERT_SQL = """
INSERT INTO adaptive_selections (position_leg_id, regime_idx, config_idx, selected_at)
VALUES (%s, %s, %s, %s)
"""

_SELECT_BY_PK_SQL = """
SELECT position_leg_id, regime_idx, config_idx, selected_at
FROM adaptive_selections
WHERE position_leg_id = %s
"""


@dataclass(slots=True)
class PgAdaptiveSelectionRepository:
    """Persists adaptive selection events into PostgreSQL."""

    connection: DbConnection

    async def record_selection(self, event: AdaptiveSelectionEvent) -> None:
        """Persist an arm selection event."""
        row = (
            event.position_leg_id,
            event.arm.regime_idx,
            event.arm.config_idx,
            event.selected_at.isoformat(),
        )
        await pg_execute(self.connection, _INSERT_SQL, row, error_msg="failed to record adaptive selection")

    async def get_selection(self, position_leg_id: int) -> AdaptiveSelectionEvent | None:
        """Return the selection event for a position leg, or None if absent."""
        row = await pg_fetch_one(
            self.connection, _SELECT_BY_PK_SQL, (position_leg_id,), error_msg="failed to get adaptive selection"
        )
        if row is None:
            return None
        (leg_id, regime_idx, config_idx, selected_at) = row
        return AdaptiveSelectionEvent(
            position_leg_id=int(str(leg_id)),
            arm=AdaptiveArmKey(regime_idx=int(str(regime_idx)), config_idx=int(str(config_idx))),
            selected_at=parse_pg_datetime(selected_at),
        )

"""PostgreSQL adaptive outcome repository."""

from dataclasses import dataclass
from datetime import UTC, datetime

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveOutcomeEvent

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO adaptive_outcomes (selection_id, result, reward, recorded_at)
VALUES (%s, %s::adaptive_outcome_result, %s, %s)
"""

_SELECT_BY_POSITION_SQL = """
SELECT ao.selection_id, s.regime_idx, s.config_idx, ao.reward, ao.recorded_at
FROM adaptive_outcomes ao
JOIN adaptive_selections s ON s.position_leg_id = ao.selection_id
WHERE ao.selection_id = %s
"""


def _reward_to_result(reward: float) -> str:
    """Map reward to result enum value."""
    if reward > 0.5:
        return "win"
    if reward < 0.5:
        return "loss"
    return "breakeven"


@dataclass(slots=True)
class PgAdaptiveOutcomeRepository:
    """Persists adaptive outcome events into PostgreSQL."""

    connection: DbConnection

    async def record_outcome(self, event: AdaptiveOutcomeEvent) -> None:
        """Persist an outcome event."""
        result = _reward_to_result(event.reward)
        row = (event.position_leg_id, result, event.reward, event.observed_at.isoformat())
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to record adaptive outcome: {exc}") from exc

    async def get_outcome(self, position_leg_id: int) -> AdaptiveOutcomeEvent | None:
        """Return the outcome event for a position, or None if absent."""
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_POSITION_SQL, (position_leg_id,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get adaptive outcome: {exc}") from exc
        if row is None:
            return None
        (selection_id, regime_idx, config_idx, reward, recorded_at) = row
        if isinstance(recorded_at, str):
            recorded_at = datetime.fromisoformat(recorded_at)
        if isinstance(recorded_at, datetime) and recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=UTC)
        if not isinstance(recorded_at, datetime):
            raise AdapterError("adaptive_outcome.recorded_at is not a datetime")
        return AdaptiveOutcomeEvent(
            position_leg_id=int(str(selection_id)),
            arm=AdaptiveArmKey(regime_idx=int(str(regime_idx)), config_idx=int(str(config_idx))),
            reward=float(str(reward)),
            observed_at=recorded_at,
        )

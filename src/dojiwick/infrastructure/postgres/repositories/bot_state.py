"""PostgreSQL bot state repository."""

from dataclasses import dataclass

from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.models.entities.bot_state import BotState
from dojiwick.domain.errors import AdapterError

from dojiwick.infrastructure.postgres.connection import DbConnection

_GET_SQL = """
SELECT consecutive_errors, consecutive_losses, daily_trade_count, daily_pnl_usd,
       circuit_breaker_active, circuit_breaker_until, last_tick_at, last_decay_at,
       daily_reset_at, recon_health, recon_health_since, recon_frozen_symbols
FROM bot_state
WHERE id = 1
"""

_ENSURE_SQL = """
INSERT INTO bot_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING
"""

_UPDATE_SQL = """
UPDATE bot_state SET
    consecutive_errors = %s,
    consecutive_losses = %s,
    daily_trade_count = %s,
    daily_pnl_usd = %s,
    circuit_breaker_active = %s,
    circuit_breaker_until = %s,
    last_tick_at = %s,
    last_decay_at = %s,
    daily_reset_at = %s,
    recon_health = %s,
    recon_health_since = %s,
    recon_frozen_symbols = %s,
    updated_at = now()
WHERE id = 1
"""

_INSERT_HISTORY_SQL = """
INSERT INTO bot_state_history (
    consecutive_errors, consecutive_losses, daily_trade_count, daily_pnl_usd,
    circuit_breaker_active, circuit_breaker_until, last_tick_at,
    recon_health, recon_frozen_symbols
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(slots=True)
class PgBotStateRepository:
    """Persists circuit breaker state into PostgreSQL."""

    connection: DbConnection

    async def get_state(self) -> BotState:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_ENSURE_SQL)
                await cursor.execute(_GET_SQL)
                row = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get bot state: {exc}") from exc
        if row is None:
            return BotState()
        return BotState(
            consecutive_errors=row[0],
            consecutive_losses=row[1],
            daily_trade_count=row[2],
            daily_pnl_usd=row[3],
            circuit_breaker_active=row[4],
            circuit_breaker_until=row[5],
            last_tick_at=row[6],
            last_decay_at=row[7],
            daily_reset_at=row[8],
            recon_health=ReconciliationHealth(row[9]),
            recon_health_since=row[10],
            recon_frozen_symbols=tuple(row[11]) if row[11] else (),
        )

    async def update_state(self, state: BotState) -> None:
        row = (
            state.consecutive_errors,
            state.consecutive_losses,
            state.daily_trade_count,
            state.daily_pnl_usd,
            state.circuit_breaker_active,
            state.circuit_breaker_until,
            state.last_tick_at,
            state.last_decay_at,
            state.daily_reset_at,
            state.recon_health.value,
            state.recon_health_since,
            list(state.recon_frozen_symbols),
        )
        history_row = (
            state.consecutive_errors,
            state.consecutive_losses,
            state.daily_trade_count,
            state.daily_pnl_usd,
            state.circuit_breaker_active,
            state.circuit_breaker_until,
            state.last_tick_at,
            state.recon_health.value,
            list(state.recon_frozen_symbols),
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_INSERT_HISTORY_SQL, history_row)
                await cursor.execute(_UPDATE_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to update bot state: {exc}") from exc

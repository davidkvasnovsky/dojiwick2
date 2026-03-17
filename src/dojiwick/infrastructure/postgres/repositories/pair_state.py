"""PostgreSQL pair state repository."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.entities.pair_state import PairTradingState

from dojiwick.infrastructure.postgres.connection import DbConnection

_GET_SQL = """
SELECT pair, target_id, wins, losses, consecutive_losses, last_trade_at, blocked, venue, product
FROM pair_trading_state
WHERE pair = %s
"""

_GET_ALL_SQL = """
SELECT pair, target_id, wins, losses, consecutive_losses, last_trade_at, blocked, venue, product
FROM pair_trading_state
ORDER BY pair
"""

_UPSERT_SQL = """
INSERT INTO pair_trading_state (pair, target_id, venue, product, wins, losses, consecutive_losses, last_trade_at, blocked, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (pair) DO UPDATE SET
    target_id = EXCLUDED.target_id,
    venue = EXCLUDED.venue,
    product = EXCLUDED.product,
    wins = EXCLUDED.wins,
    losses = EXCLUDED.losses,
    consecutive_losses = EXCLUDED.consecutive_losses,
    last_trade_at = EXCLUDED.last_trade_at,
    blocked = EXCLUDED.blocked,
    updated_at = now()
"""


@dataclass(slots=True)
class PgPairStateRepository:
    """Persists per-pair trading state into PostgreSQL."""

    connection: DbConnection

    async def get_state(self, pair: str) -> PairTradingState | None:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_GET_SQL, (pair,))
                row = await cursor.fetchone()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get pair state: {exc}") from exc
        if row is None:
            return None
        return self._row_to_state(row)

    async def get_all(self) -> tuple[PairTradingState, ...]:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_GET_ALL_SQL)
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get all pair states: {exc}") from exc
        return tuple(self._row_to_state(row) for row in rows)

    async def upsert(self, state: PairTradingState) -> None:
        row = (
            state.pair,
            state.target_id,
            state.venue,
            state.product,
            state.wins,
            state.losses,
            state.consecutive_losses,
            state.last_trade_at,
            state.blocked,
        )
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_UPSERT_SQL, row)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to upsert pair state: {exc}") from exc

    @staticmethod
    def _row_to_state(row: tuple[Any, ...]) -> PairTradingState:
        pair, target_id, wins, losses, consecutive_losses, last_trade_at, blocked, venue, product = row
        if isinstance(last_trade_at, str):
            last_trade_at = datetime.fromisoformat(last_trade_at)
        if isinstance(last_trade_at, datetime) and last_trade_at.tzinfo is None:
            last_trade_at = last_trade_at.replace(tzinfo=UTC)
        return PairTradingState(
            pair=str(pair),
            target_id=str(target_id),
            venue=str(venue),
            product=str(product),
            wins=int(wins),
            losses=int(losses),
            consecutive_losses=int(consecutive_losses),
            last_trade_at=last_trade_at,
            blocked=bool(blocked),
        )

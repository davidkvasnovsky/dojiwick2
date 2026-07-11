"""PostgreSQL position exit-state repository."""

from dataclasses import dataclass

from dojiwick.domain.models.entities.position_exit_state import PositionExitState

from dojiwick.infrastructure.postgres.connection import DbConnection
from dojiwick.infrastructure.postgres.helpers import pg_execute, pg_fetch_all, pg_fetch_one

_UPSERT_SQL = """
INSERT INTO position_exit_state (
    position_leg_id, is_long, entry_price, stop_price, original_stop, take_profit_price,
    trailing_activation_price, trailing_distance, breakeven_price, extreme_price,
    max_hold_bars, bars_held, tp1_price, tp1_fraction, tp1_filled, revision
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (position_leg_id) DO UPDATE SET
    is_long = EXCLUDED.is_long, entry_price = EXCLUDED.entry_price,
    stop_price = EXCLUDED.stop_price, original_stop = EXCLUDED.original_stop,
    take_profit_price = EXCLUDED.take_profit_price,
    trailing_activation_price = EXCLUDED.trailing_activation_price,
    trailing_distance = EXCLUDED.trailing_distance, breakeven_price = EXCLUDED.breakeven_price,
    extreme_price = EXCLUDED.extreme_price, max_hold_bars = EXCLUDED.max_hold_bars,
    bars_held = EXCLUDED.bars_held, tp1_price = EXCLUDED.tp1_price,
    tp1_fraction = EXCLUDED.tp1_fraction, tp1_filled = EXCLUDED.tp1_filled,
    revision = EXCLUDED.revision, updated_at = now()
"""

_SELECT_COLUMNS = """
    position_leg_id, is_long, entry_price, stop_price, original_stop, take_profit_price,
    trailing_activation_price, trailing_distance, breakeven_price, extreme_price,
    max_hold_bars, bars_held, tp1_price, tp1_fraction, tp1_filled, revision
"""

_GET_SQL = f"SELECT {_SELECT_COLUMNS} FROM position_exit_state WHERE position_leg_id = %s"

_LIST_ACTIVE_SQL = f"""
SELECT {_SELECT_COLUMNS}
FROM position_exit_state s
JOIN position_legs l ON l.id = s.position_leg_id
WHERE l.account = %s AND l.closed_at IS NULL
"""


def _row_to_state(row: tuple[object, ...]) -> PositionExitState:
    return PositionExitState(
        position_leg_id=int(str(row[0])),
        is_long=bool(row[1]),
        entry_price=float(str(row[2])),
        stop_price=float(str(row[3])),
        original_stop=float(str(row[4])),
        take_profit_price=float(str(row[5])),
        trailing_activation_price=float(str(row[6])),
        trailing_distance=float(str(row[7])),
        breakeven_price=float(str(row[8])),
        extreme_price=float(str(row[9])),
        max_hold_bars=int(str(row[10])),
        bars_held=int(str(row[11])),
        tp1_price=float(str(row[12])),
        tp1_fraction=float(str(row[13])),
        tp1_filled=bool(row[14]),
        revision=int(str(row[15])),
    )


@dataclass(slots=True)
class PgPositionExitStateRepository:
    """Persists live exit-management state into PostgreSQL."""

    connection: DbConnection

    async def upsert(self, state: PositionExitState) -> None:
        row = (
            state.position_leg_id,
            state.is_long,
            state.entry_price,
            state.stop_price,
            state.original_stop,
            state.take_profit_price,
            state.trailing_activation_price,
            state.trailing_distance,
            state.breakeven_price,
            state.extreme_price,
            state.max_hold_bars,
            state.bars_held,
            state.tp1_price,
            state.tp1_fraction,
            state.tp1_filled,
            state.revision,
        )
        await pg_execute(self.connection, _UPSERT_SQL, row, error_msg="failed to upsert exit state")

    async def get(self, position_leg_id: int) -> PositionExitState | None:
        row = await pg_fetch_one(self.connection, _GET_SQL, (position_leg_id,), error_msg="failed to get exit state")
        return _row_to_state(row) if row is not None else None

    async def list_active(self, account: str) -> tuple[PositionExitState, ...]:
        rows = await pg_fetch_all(self.connection, _LIST_ACTIVE_SQL, (account,), error_msg="failed to list exit states")
        return tuple(_row_to_state(r) for r in rows)

    async def delete(self, position_leg_id: int) -> None:
        await pg_execute(
            self.connection,
            "DELETE FROM position_exit_state WHERE position_leg_id = %s",
            (position_leg_id,),
            error_msg="failed to delete exit state",
        )

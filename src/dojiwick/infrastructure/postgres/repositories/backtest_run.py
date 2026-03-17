"""PostgreSQL backtest run repository."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.backtest_run import BacktestRunRecord
from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary
from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO backtest_runs (
    config_hash, start_date, end_date, timeframe, pairs,
    target_ids, venue, product,
    trades, total_pnl_usd, win_rate, expectancy_usd, sharpe_like,
    max_drawdown_pct, sortino, calmar, profit_factor,
    max_consecutive_losses, payoff_ratio, source, params_json
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_SELECT_BY_HASH_SQL = """
SELECT config_hash, start_date, end_date, timeframe, pairs,
       target_ids, venue, product,
       trades, total_pnl_usd, win_rate, expectancy_usd, sharpe_like,
       max_drawdown_pct, sortino, calmar, profit_factor,
       max_consecutive_losses, payoff_ratio, source, params_json
FROM backtest_runs
WHERE config_hash = %s
ORDER BY created_at DESC
"""


@dataclass(slots=True)
class PgBacktestRunRepository:
    """Persists backtest run records into PostgreSQL."""

    connection: DbConnection

    async def insert(self, record: BacktestRunRecord) -> int:
        s = record.summary
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(
                    _INSERT_SQL,
                    (
                        record.config_hash,
                        record.start_date.isoformat(),
                        record.end_date.isoformat(),
                        record.interval,
                        list(record.pairs),
                        list(record.target_ids),
                        record.venue,
                        record.product,
                        s.trades,
                        s.total_pnl_usd,
                        s.win_rate,
                        s.expectancy_usd,
                        s.sharpe_like,
                        s.max_drawdown_pct,
                        s.sortino,
                        s.calmar,
                        s.profit_factor,
                        s.max_consecutive_losses,
                        s.payoff_ratio,
                        record.source,
                        json.dumps(record.params_json),
                    ),
                )
                row = await cursor.fetchone()
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert backtest run: {exc}") from exc
        assert row is not None
        return int(row[0])

    async def get_by_config_hash(self, config_hash: str) -> tuple[BacktestRunRecord, ...]:
        try:
            async with self.connection.cursor() as cursor:
                await cursor.execute(_SELECT_BY_HASH_SQL, (config_hash,))
                rows = await cursor.fetchall()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to get backtest runs: {exc}") from exc
        return tuple(
            BacktestRunRecord(
                config_hash=row[0],
                start_date=_ensure_tz(row[1]),
                end_date=_ensure_tz(row[2]),
                interval=row[3],
                pairs=tuple(row[4]),
                target_ids=tuple(row[5]),
                venue=row[6],
                product=row[7],
                summary=BacktestSummary(
                    trades=row[8],
                    total_pnl_usd=float(row[9]),
                    win_rate=row[10],
                    expectancy_usd=float(row[11]),
                    sharpe_like=row[12],
                    max_drawdown_pct=row[13],
                    sortino=row[14],
                    calmar=row[15],
                    profit_factor=row[16],
                    max_consecutive_losses=row[17],
                    payoff_ratio=row[18],
                ),
                source=row[19],
                params_json=_parse_params_json(row[20]),
            )
            for row in rows
        )


def _parse_params_json(raw: dict[str, object] | str) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    return cast(dict[str, object], json.loads(str(raw)))


def _ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt

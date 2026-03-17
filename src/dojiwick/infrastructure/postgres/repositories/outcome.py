"""PostgreSQL outcome repository."""

from dataclasses import dataclass

from dojiwick.domain.enums import MARKET_STATE_TO_SQL, TRADE_ACTION_TO_SQL
from dojiwick.domain.errors import AdapterError
from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO decision_outcomes (
    venue, product, target_id,
    pair, observed_at, status, authority, reason_code,
    action, strategy_name, strategy_variant, confidence,
    entry_price, stop_price, take_profit_price,
    quantity, notional_usd, config_hash, order_id, note, market_state,
    tick_id, confidence_raw
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(slots=True)
class PgOutcomeRepository:
    """Persists outcome batches into PostgreSQL."""

    connection: DbConnection

    async def append_outcomes(self, outcomes: tuple[DecisionOutcome, ...], *, venue: str, product: str) -> None:
        """Insert all outcomes in one transaction."""
        if not venue or not product:
            raise AdapterError("append_outcomes requires non-empty venue and product")

        rows = [
            (
                venue,
                product,
                outcome.target_id,
                outcome.pair,
                outcome.observed_at.isoformat(),
                outcome.status.value,
                outcome.authority.value,
                outcome.reason_code,
                TRADE_ACTION_TO_SQL[outcome.action.value],
                outcome.strategy_name,
                outcome.strategy_variant,
                outcome.confidence,
                outcome.entry_price,
                outcome.stop_price,
                outcome.take_profit_price,
                outcome.quantity,
                outcome.notional_usd,
                outcome.config_hash,
                outcome.order_id,
                outcome.note,
                MARKET_STATE_TO_SQL[outcome.market_state.value],
                outcome.tick_id,
                outcome.confidence_raw,
            )
            for outcome in outcomes
        ]

        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_INSERT_SQL, rows)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to append outcomes: {exc}") from exc

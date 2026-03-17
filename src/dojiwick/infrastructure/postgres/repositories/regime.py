"""PostgreSQL regime observation repository."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile
from dojiwick.domain.errors import AdapterError

from dojiwick.infrastructure.postgres.connection import DbConnection

_INSERT_SQL = """
INSERT INTO regime_observations (
    pair,
    observed_at,
    coarse_state,
    confidence,
    valid,
    target_id,
    venue,
    product
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(slots=True)
class PgRegimeRepository:
    """Persists regime batches into PostgreSQL."""

    connection: DbConnection

    async def insert_batch(
        self,
        pairs: tuple[str, ...],
        observed_at: datetime,
        regimes: BatchRegimeProfile,
        *,
        target_ids: tuple[str, ...],
        venue: str,
        product: str,
    ) -> None:
        """Insert one classified regime batch."""
        if not venue or not product:
            raise AdapterError("insert_batch requires non-empty venue and product")
        if len(target_ids) != len(pairs):
            raise AdapterError(f"target_ids length ({len(target_ids)}) must match pairs length ({len(pairs)})")

        rows = [
            (
                pair,
                observed_at.isoformat(),
                int(regimes.coarse_state[index]),
                float(regimes.confidence[index]),
                bool(regimes.valid_mask[index]),
                target_ids[index],
                venue,
                product,
            )
            for index, pair in enumerate(pairs)
        ]

        try:
            async with self.connection.cursor() as cursor:
                await cursor.executemany(_INSERT_SQL, rows)
            await self.connection.commit()
        except Exception as exc:
            await self.connection.rollback()
            raise AdapterError(f"failed to insert regime batch: {exc}") from exc

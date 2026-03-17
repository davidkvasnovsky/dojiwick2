"""PostgreSQL adaptive posterior repository."""

from dataclasses import dataclass

from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptivePosterior
from dojiwick.infrastructure.postgres.connection import DbConnection
from dojiwick.infrastructure.postgres.helpers import parse_pg_datetime_optional, pg_execute, pg_fetch_all

_SELECT_BY_REGIME_SQL = """
SELECT regime_idx, config_idx, alpha, beta, n_updates, last_decay_at
FROM adaptive_posteriors
WHERE regime_idx = %s
ORDER BY config_idx
"""

_UPSERT_SQL = """
INSERT INTO adaptive_posteriors (regime_idx, config_idx, alpha, beta, n_updates, last_decay_at)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (regime_idx, config_idx) DO UPDATE
SET alpha = EXCLUDED.alpha,
    beta = EXCLUDED.beta,
    n_updates = EXCLUDED.n_updates,
    last_decay_at = EXCLUDED.last_decay_at
"""


@dataclass(slots=True)
class PgAdaptivePosteriorRepository:
    """Persists adaptive posteriors into PostgreSQL."""

    connection: DbConnection

    async def get_posteriors(self, regime_idx: int) -> tuple[AdaptivePosterior, ...]:
        """Return all posteriors for a given regime."""
        rows = await pg_fetch_all(
            self.connection, _SELECT_BY_REGIME_SQL, (regime_idx,), error_msg="failed to get adaptive posteriors"
        )
        results: list[AdaptivePosterior] = []
        for row in rows:
            (regime, config, alpha, beta, n_updates, last_decay_at) = row
            results.append(
                AdaptivePosterior(
                    arm=AdaptiveArmKey(regime_idx=int(str(regime)), config_idx=int(str(config))),
                    alpha=float(str(alpha)),
                    beta=float(str(beta)),
                    n_updates=int(str(n_updates)),
                    last_decay_at=parse_pg_datetime_optional(last_decay_at),
                )
            )
        return tuple(results)

    async def upsert_posterior(self, posterior: AdaptivePosterior) -> None:
        """Insert or update a posterior."""
        decay_iso = posterior.last_decay_at.isoformat() if posterior.last_decay_at else None
        row = (
            posterior.arm.regime_idx,
            posterior.arm.config_idx,
            posterior.alpha,
            posterior.beta,
            posterior.n_updates,
            decay_iso,
        )
        await pg_execute(self.connection, _UPSERT_SQL, row, error_msg="failed to upsert adaptive posterior")

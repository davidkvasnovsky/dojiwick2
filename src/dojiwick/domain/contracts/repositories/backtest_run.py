"""Backtest run repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.backtest_run import BacktestRunRecord


class BacktestRunRepositoryPort(Protocol):
    """Persists backtest run results."""

    async def insert(self, record: BacktestRunRecord) -> int:
        """Insert a backtest run record. Returns the generated ID."""
        ...

    async def get_by_config_hash(self, config_hash: str) -> tuple[BacktestRunRecord, ...]:
        """Return all runs matching a config hash."""
        ...

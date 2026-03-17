"""Backtest run repository test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.backtest_run import BacktestRunRecord


@dataclass(slots=True)
class InMemoryBacktestRunRepo:
    """In-memory backtest run store for tests."""

    _records: list[BacktestRunRecord] = field(default_factory=list)
    _next_id: int = 1

    async def insert(self, record: BacktestRunRecord) -> int:
        self._records.append(record)
        run_id = self._next_id
        self._next_id += 1
        return run_id

    async def get_by_config_hash(self, config_hash: str) -> tuple[BacktestRunRecord, ...]:
        return tuple(r for r in self._records if r.config_hash == config_hash)

    @property
    def records(self) -> list[BacktestRunRecord]:
        return self._records

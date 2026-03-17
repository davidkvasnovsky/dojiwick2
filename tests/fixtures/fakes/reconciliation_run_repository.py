"""Fake reconciliation run repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.reconciliation_run import ReconciliationRun


@dataclass(slots=True)
class FakeReconciliationRunRepository:
    """In-memory reconciliation run repository for test assertions."""

    runs: list[ReconciliationRun] = field(default_factory=list)
    _next_id: int = field(default=1, init=False, repr=False)

    async def insert_run(self, run: ReconciliationRun) -> int:
        run_id = self._next_id
        self._next_id += 1
        self.runs.append(run)
        return run_id

    async def get_latest(self, run_type: str | None = None) -> ReconciliationRun | None:
        filtered = self.runs if run_type is None else [r for r in self.runs if r.run_type == run_type]
        return filtered[-1] if filtered else None

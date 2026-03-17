"""Execution planner port test doubles."""

from dojiwick.domain.models.value_objects.account_state import AccountSnapshot
from dojiwick.domain.models.value_objects.exchange_types import TargetLegPosition
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan


class FakeExecutionPlanner:
    """In-memory fake for ExecutionPlannerPort — returns configurable execution plans."""

    def __init__(self, default_plan: ExecutionPlan | None = None) -> None:
        self._default_plan = default_plan
        self._calls: list[tuple[AccountSnapshot, tuple[TargetLegPosition, ...]]] = []

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Test helper: set the plan to return on next call."""
        self._default_plan = plan

    @property
    def calls(self) -> list[tuple[AccountSnapshot, tuple[TargetLegPosition, ...]]]:
        """Test helper: return captured plan() call arguments."""
        return self._calls

    async def plan(
        self,
        account_snapshot: AccountSnapshot,
        targets: tuple[TargetLegPosition, ...],
    ) -> ExecutionPlan:
        self._calls.append((account_snapshot, targets))
        if self._default_plan is not None:
            return self._default_plan
        return ExecutionPlan(account=account_snapshot.account, deltas=())

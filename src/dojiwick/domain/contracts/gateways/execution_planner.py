"""Execution planner port — computing leg deltas from current state + target positions."""

from typing import Protocol

from dojiwick.domain.models.value_objects.account_state import AccountSnapshot
from dojiwick.domain.models.value_objects.exchange_types import TargetLegPosition
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan


class ExecutionPlannerPort(Protocol):
    """Computes leg deltas from current state + target positions, returning an execution plan."""

    async def plan(
        self,
        account_snapshot: AccountSnapshot,
        targets: tuple[TargetLegPosition, ...],
    ) -> ExecutionPlan:
        """Compute the execution plan to move from current positions to targets."""
        ...

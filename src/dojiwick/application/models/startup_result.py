"""Startup orchestrator result value object."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dojiwick.application.use_cases.run_reconciliation import ReconciliationService
from dojiwick.domain.models.value_objects.health_state import HealthState


@dataclass(slots=True, frozen=True, kw_only=True)
class StartupResult:
    """Captures the output of the startup orchestrator."""

    health: HealthState
    consumer_task: asyncio.Task[None] | None = None
    periodic_reconciliation: ReconciliationService | None = None
    cancelled_orders: int = 0
    replayed_events: int = 0

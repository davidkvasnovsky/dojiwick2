"""Daily LLM cost tracking with budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort
from dojiwick.domain.models.value_objects.model_cost import ModelCostRecord

if TYPE_CHECKING:
    from dojiwick.domain.contracts.repositories.model_cost import ModelCostRepositoryPort


@dataclass(slots=True)
class CostTracker:
    """Tracks daily LLM spend and enforces budget limits."""

    daily_budget_usd: float
    clock: ClockPort
    input_cost_per_token: float
    output_cost_per_token: float
    metrics: MetricsSinkPort | None = None
    cost_repository: ModelCostRepositoryPort | None = None
    current_tick_id: str = ""
    _day_start_date: str = field(default="", init=False, repr=False)
    _day_spend_usd: float = field(default=0.0, init=False, repr=False)
    _pending: list[ModelCostRecord] = field(default_factory=list, init=False, repr=False)

    async def record(self, input_tokens: int, output_tokens: int, *, model: str = "", purpose: str = "") -> None:
        """Record token usage and accumulate cost."""
        self._maybe_reset()
        cost = (input_tokens * self.input_cost_per_token) + (output_tokens * self.output_cost_per_token)
        self._day_spend_usd += cost
        if self.metrics is not None:
            self.metrics.gauge("ai_cost_usd_daily", self._day_spend_usd)
        if self.cost_repository is not None and self.current_tick_id:
            self._pending.append(
                ModelCostRecord(
                    tick_id=self.current_tick_id,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=Decimal(str(cost)),
                    purpose=purpose,
                    created_at=self.clock.now_utc(),
                )
            )

    async def flush(self) -> None:
        """Flush pending cost records to the repository in a single batch."""
        if not self._pending or self.cost_repository is None:
            return
        await self.cost_repository.batch_record_costs(tuple(self._pending))
        self._pending.clear()

    @property
    def day_spend_usd(self) -> float:
        """Return current day's accumulated spend."""
        self._maybe_reset()
        return self._day_spend_usd

    def is_budget_exceeded(self) -> bool:
        """Check if daily budget has been exceeded."""
        self._maybe_reset()
        return self._day_spend_usd >= self.daily_budget_usd

    def _maybe_reset(self) -> None:
        today = self.clock.now_utc().strftime("%Y-%m-%d")
        if today != self._day_start_date:
            self._day_start_date = today
            self._day_spend_usd = 0.0

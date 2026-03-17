"""Tests for the cost tracker."""

from datetime import UTC, datetime

from dojiwick.infrastructure.ai.cost_tracker import CostTracker
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.model_cost_repository import FakeModelCostRepository


class TestCostTracker:
    async def test_record_accumulates_cost(self) -> None:
        clock = FixedClock()
        tracker = CostTracker(
            daily_budget_usd=1.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        await tracker.record(input_tokens=1000, output_tokens=100)
        assert not tracker.is_budget_exceeded()

    async def test_budget_exceeded(self) -> None:
        clock = FixedClock()
        # Sonnet: $3/MTok input, $15/MTok output
        # 1M input tokens = $3, exceeds $1 budget
        tracker = CostTracker(
            daily_budget_usd=1.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        await tracker.record(input_tokens=1_000_000, output_tokens=0)
        assert tracker.is_budget_exceeded()

    async def test_daily_reset(self) -> None:
        clock = FixedClock(at=datetime(2026, 1, 1, 23, 59, tzinfo=UTC))
        tracker = CostTracker(
            daily_budget_usd=1.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        await tracker.record(input_tokens=1_000_000, output_tokens=0)
        assert tracker.is_budget_exceeded()

        # Advance to next day
        clock.set(datetime(2026, 1, 2, 0, 1, tzinfo=UTC))
        assert not tracker.is_budget_exceeded()

    async def test_uses_injected_clock(self) -> None:
        clock = FixedClock(at=datetime(2026, 3, 15, 12, 0, tzinfo=UTC))
        tracker = CostTracker(
            daily_budget_usd=10.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        await tracker.record(input_tokens=100, output_tokens=50)
        assert not tracker.is_budget_exceeded()

    async def test_injected_pricing_applies_to_any_model(self) -> None:
        clock = FixedClock()
        tracker = CostTracker(
            daily_budget_usd=10.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
        )
        await tracker.record(input_tokens=1000, output_tokens=100)
        # $3/MTok * 1000 + $15/MTok * 100 = $0.003 + $0.0015 = $0.0045 — well under $10
        assert not tracker.is_budget_exceeded()

    async def test_flush_batches_pending_records(self) -> None:
        clock = FixedClock()
        repo = FakeModelCostRepository()
        tracker = CostTracker(
            daily_budget_usd=10.0,
            clock=clock,
            input_cost_per_token=3.0 / 1_000_000,
            output_cost_per_token=15.0 / 1_000_000,
            cost_repository=repo,
            current_tick_id="tick-1",
        )
        await tracker.record(input_tokens=1000, output_tokens=100, model="sonnet", purpose="veto")
        await tracker.record(input_tokens=2000, output_tokens=200, model="sonnet", purpose="veto")
        # Records are buffered, not yet flushed
        assert len(repo.costs) == 0
        await tracker.flush()
        assert len(repo.costs) == 2
        # Second flush is a no-op
        await tracker.flush()
        assert len(repo.costs) == 2

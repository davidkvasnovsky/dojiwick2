"""Tests for decision trace collection in TickService."""

import pytest

from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace
from fixtures.fakes.decision_trace_repository import InMemoryDecisionTraceRepository


class TestDecisionTrace:
    def test_frozen(self) -> None:
        trace = DecisionTrace(tick_id="t1", step_name="pipeline", step_seq=0)
        with pytest.raises(AttributeError):
            trace.step_name = "x"  # type: ignore[misc]

    def test_default_artifacts(self) -> None:
        trace = DecisionTrace(tick_id="t1", step_name="pipeline", step_seq=0)
        assert trace.artifacts == {}
        assert trace.step_hash == ""
        assert trace.duration_us is None


class TestInMemoryDecisionTraceRepository:
    @pytest.mark.asyncio
    async def test_insert_batch(self) -> None:
        repo = InMemoryDecisionTraceRepository()
        traces = (
            DecisionTrace(tick_id="t1", step_name="pipeline", step_seq=0, duration_us=100),
            DecisionTrace(tick_id="t1", step_name="execution", step_seq=1, duration_us=200),
            DecisionTrace(tick_id="t1", step_name="persistence", step_seq=2, duration_us=50),
        )
        await repo.insert_batch(traces)
        assert len(repo.traces) == 3
        assert repo.traces[0].step_name == "pipeline"
        assert repo.traces[1].step_seq == 1
        assert repo.traces[2].duration_us == 50

    @pytest.mark.asyncio
    async def test_insert_empty(self) -> None:
        repo = InMemoryDecisionTraceRepository()
        await repo.insert_batch(())
        assert repo.traces == []

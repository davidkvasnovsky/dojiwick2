"""In-memory decision trace repository for testing."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace


@dataclass(slots=True)
class InMemoryDecisionTraceRepository:
    """Captures decision traces in a list for test assertions."""

    traces: list[DecisionTrace] = field(default_factory=list)

    async def insert_batch(self, traces: tuple[DecisionTrace, ...]) -> None:
        self.traces.extend(traces)

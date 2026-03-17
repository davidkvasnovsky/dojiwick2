"""Decision trace repository port."""

from typing import Protocol

from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace


class DecisionTraceRepositoryPort(Protocol):
    """Persists decision trace records."""

    async def insert_batch(self, traces: tuple[DecisionTrace, ...]) -> None:
        """Insert a batch of decision traces."""
        ...

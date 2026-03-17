"""Model cost repository protocol."""

from typing import Protocol

from dojiwick.domain.models.value_objects.model_cost import ModelCostRecord


class ModelCostRepositoryPort(Protocol):
    """LLM cost persistence."""

    async def record_cost(self, record: ModelCostRecord) -> None:
        """Persist a model cost entry."""
        ...

    async def batch_record_costs(self, records: tuple[ModelCostRecord, ...]) -> None:
        """Persist multiple model cost entries in a single batch."""
        ...

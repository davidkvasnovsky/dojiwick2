"""Fake model cost repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.value_objects.model_cost import ModelCostRecord


@dataclass(slots=True)
class FakeModelCostRepository:
    """In-memory model cost repository for test assertions."""

    costs: list[ModelCostRecord] = field(default_factory=list)

    async def record_cost(self, record: ModelCostRecord) -> None:
        self.costs.append(record)

    async def batch_record_costs(self, records: tuple[ModelCostRecord, ...]) -> None:
        self.costs.extend(records)

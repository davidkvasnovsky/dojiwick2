"""Model cost record for LLM token usage tracking."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCostRecord:
    """A single LLM cost entry stored in the model_costs table."""

    tick_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    purpose: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

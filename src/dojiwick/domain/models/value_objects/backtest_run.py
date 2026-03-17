"""Backtest run record for persistence."""

from dataclasses import dataclass, field
from datetime import datetime

from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary


@dataclass(slots=True, frozen=True, kw_only=True)
class BacktestRunRecord:
    """Immutable record of a completed backtest run."""

    config_hash: str
    start_date: datetime
    end_date: datetime
    interval: str
    pairs: tuple[str, ...]
    target_ids: tuple[str, ...]
    venue: str
    product: str
    summary: BacktestSummary
    source: str = "backtest"
    params_json: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_hash:
            raise ValueError("config_hash must not be empty")
        if self.start_date.tzinfo is None:
            raise ValueError("start_date must be timezone-aware")
        if self.end_date.tzinfo is None:
            raise ValueError("end_date must be timezone-aware")
        if not self.pairs:
            raise ValueError("pairs must not be empty")
        if not self.target_ids:
            raise ValueError("target_ids must not be empty")
        if not self.venue:
            raise ValueError("venue must not be empty")
        if not self.product:
            raise ValueError("product must not be empty")
        if len(self.target_ids) != len(self.pairs):
            raise ValueError(f"target_ids length ({len(self.target_ids)}) must match pairs length ({len(self.pairs)})")

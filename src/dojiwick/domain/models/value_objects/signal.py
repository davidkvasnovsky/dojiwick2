"""Signal domain model."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class Signal:
    """A detected market event signal."""

    pair: str
    target_id: str
    signal_type: str
    priority: int = 0
    details: dict[str, object] | None = None
    detected_at: datetime | None = None
    decision_outcome_id: int | None = None

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must not be empty")
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if not self.signal_type:
            raise ValueError("signal_type must not be empty")
        if self.detected_at is not None and self.detected_at.tzinfo is None:
            raise ValueError("detected_at must be timezone-aware")

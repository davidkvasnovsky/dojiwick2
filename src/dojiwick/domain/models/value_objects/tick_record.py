"""Tick lifecycle record for determinism tracking and deduplication."""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.enums import DecisionAuthority, TickStatus


@dataclass(slots=True, frozen=True, kw_only=True)
class TickRecord:
    """Immutable snapshot of a single tick's lifecycle."""

    tick_id: str
    tick_time: datetime
    config_hash: str
    schema_ver: int = 1
    inputs_hash: str
    intent_hash: str = ""
    ops_hash: str = ""
    authority: DecisionAuthority = DecisionAuthority.DETERMINISTIC_ONLY
    status: TickStatus = TickStatus.STARTED
    batch_size: int = 0
    duration_ms: int | None = None
    error_message: str | None = None
